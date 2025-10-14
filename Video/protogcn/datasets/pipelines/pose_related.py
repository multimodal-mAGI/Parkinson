import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import mode as get_mode

from ..builder import PIPELINES
from .compose import Compose
from .formatting import Rename

import sys

EPS = 1e-4


@PIPELINES.register_module()
class PoseDecode:
    """Load and decode pose with given indices.

    Required keys are "keypoint", "frame_inds" (optional), "keypoint_score" (optional), added or modified keys are
    "keypoint", "keypoint_score" (if applicable).
    """

    @staticmethod
    def _load_kp(kp, frame_inds):
        return kp[:, frame_inds].astype(np.float32)

    @staticmethod
    def _load_kpscore(kpscore, frame_inds):
        return kpscore[:, frame_inds].astype(np.float32)

    def __call__(self, results):

        if 'frame_inds' not in results:
            results['frame_inds'] = np.arange(results['total_frames'])

        if results['frame_inds'].ndim != 1:
            results['frame_inds'] = np.squeeze(results['frame_inds'])

        offset = results.get('offset', 0)
        frame_inds = results['frame_inds'] + offset

        if 'keypoint_score' in results:
            results['keypoint_score'] = self._load_kpscore(results['keypoint_score'], frame_inds)

        if 'keypoint' in results:
            results['keypoint'] = self._load_kp(results['keypoint'], frame_inds)

        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}()'
        return repr_str


@PIPELINES.register_module()
class PreNormalize3D:
    """PreNormalize for NTURGB+D 3D keypoints (x, y, z)."""

    def unit_vector(self, vector):
        """Returns the unit vector of the vector. """
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        """Returns the angle in radians between vectors 'v1' and 'v2'. """
        if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
            return 0
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def rotation_matrix(self, axis, theta):
        """Return the rotation matrix associated with counterclockwise rotation
        about the given axis by theta radians."""
        if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
            return np.eye(3)
        axis = np.asarray(axis)
        axis = axis / np.sqrt(np.dot(axis, axis))
        a = np.cos(theta / 2.0)
        b, c, d = -axis * np.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    def __init__(self, zaxis=[0, 1], xaxis=[8, 4], align_spine=True, align_center=True):
        self.zaxis = zaxis
        self.xaxis = xaxis
        self.align_spine = align_spine
        self.align_center = align_center

    def __call__(self, results):
        skeleton = results['keypoint']
        total_frames = results.get('total_frames', skeleton.shape[1])

        M, T, V, C = skeleton.shape
        assert T == total_frames
        if skeleton.sum() == 0:
            return results

        index0 = [i for i in range(T) if not np.all(np.isclose(skeleton[0, i], 0))]

        assert M in [1, 2]
        if M == 2:
            index1 = [i for i in range(T) if not np.all(np.isclose(skeleton[1, i], 0))]
            if len(index0) < len(index1):
                skeleton = skeleton[:, np.array(index1)]
                skeleton = skeleton[[1, 0]]
            else:
                skeleton = skeleton[:, np.array(index0)]
        else:
            skeleton = skeleton[:, np.array(index0)]

        T_new = skeleton.shape[1]

        if self.align_center:
            if skeleton.shape[2] == 25:
                main_body_center = skeleton[0, 0, 1].copy()
            else:
                main_body_center = skeleton[0, 0, -1].copy()
            mask = ((skeleton != 0).sum(-1) > 0)[..., None]
            skeleton = (skeleton - main_body_center) * mask

        if self.align_spine:
            joint_bottom = skeleton[0, 0, self.zaxis[0]]
            joint_top = skeleton[0, 0, self.zaxis[1]]
            axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
            angle = self.angle_between(joint_top - joint_bottom, [0, 0, 1])
            matrix_z = self.rotation_matrix(axis, angle)
            skeleton = np.einsum('abcd,kd->abck', skeleton, matrix_z)

            joint_rshoulder = skeleton[0, 0, self.xaxis[0]]
            joint_lshoulder = skeleton[0, 0, self.xaxis[1]]
            axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
            angle = self.angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
            matrix_x = self.rotation_matrix(axis, angle)
            skeleton = np.einsum('abcd,kd->abck', skeleton, matrix_x)

        results['keypoint'] = skeleton
        results['total_frames'] = T_new
        results['body_center'] = main_body_center
        return results


@PIPELINES.register_module()
class RandomRot:

    def __init__(self, theta=0.3):
        self.theta = theta

    def _rot3d(self, theta):
        cos, sin = np.cos(theta), np.sin(theta)
        rx = np.array([[1, 0, 0], [0, cos[0], sin[0]], [0, -sin[0], cos[0]]])
        ry = np.array([[cos[1], 0, -sin[1]], [0, 1, 0], [sin[1], 0, cos[1]]])
        rz = np.array([[cos[2], sin[2], 0], [-sin[2], cos[2], 0], [0, 0, 1]])

        rot = np.matmul(rz, np.matmul(ry, rx))
        return rot

    def _rot2d(self, theta):
        cos, sin = np.cos(theta), np.sin(theta)
        return np.array([[cos, -sin], [sin, cos]])

    def __call__(self, results):
        skeleton = results['keypoint']
        M, T, V, C = skeleton.shape

        if np.all(np.isclose(skeleton, 0)):
            return results

        assert C in [2, 3]
        if C == 3:
            theta = np.random.uniform(-self.theta, self.theta, size=3)
            rot_mat = self._rot3d(theta)
        elif C == 2:
            theta = np.random.uniform(-self.theta)
            rot_mat = self._rot2d(theta)
        results['keypoint'] = np.einsum('ab,mtvb->mtva', rot_mat, skeleton)

        return results


@PIPELINES.register_module()
class Spatial_Flip:
    """Flip the skeleton. """
    
    def __init__(self, dataset='nturgb+d', p=0.5):
        assert isinstance(p, tuple) or isinstance(p, float)
        self.dataset = dataset
        self.p = p

    def __call__(self, results):
        skeleton = results['keypoint']
        p = self.p
        transform_order = {'ntu': [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 16, 17, 18, 
                                    19, 12, 13, 14, 15, 20, 23, 24, 21, 22],
                           'nw_ucla':[0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 16, 17,
                                    18, 19, 12, 13, 14, 15],
                           'openpose':[0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10,
                                    15, 14, 17, 16]
                           }
        if random.random() < p:
            if self.dataset == 'nturgb+d':
                index = transform_order['ntu']
            elif self.dataset == 'nw_ucla':
                index = transform_order['nw_ucla']
            elif self.dataset == 'openpose':
                index = transform_order['openpose']
            trans_skeleton = skeleton[:, :, index, :]
            results['keypoint'] = trans_skeleton
        else:
            results['keypoint'] = skeleton
        
        return results


@PIPELINES.register_module()
class Part_Drop:
    """Drop the left or right limbs of the skeleton. """

    def __init__(self, p=0.2):
        assert isinstance(p, tuple) or isinstance(p, float)
        self.p = p

    def __call__(self, results):
        skeleton = results['keypoint']
        p = self.p

        if random.random() < p:     
            left_hand = [4, 5, 6 ,7, 22, 21]
            left_leg = [12, 13, 14, 15]
            right_hand = [8, 9, 10, 11, 24, 23]
            right_leg = [16, 17, 18, 19]                  
            
            part = random.randint(0, 3)    
            temp = skeleton.copy()
            # M T V C -> V M T C
            temp = temp.transpose(2, 0, 1, 3) 
            M, T, V, C = skeleton.shape
            x_new = np.zeros((M, T, C))
            if part == 0:
                for idx in left_hand:
                    temp[idx] = x_new
            elif part == 1:
                for idx in left_leg:
                    temp[idx] = x_new 
            elif part == 2:
                for idx in right_hand:
                    temp[idx] = x_new
            elif part == 3:
                for idx in right_leg:
                    temp[idx] = x_new
                    
            # V M T C -> M T V C
            temp = temp.transpose(1, 2, 0, 3)
            results['keypoint'] = temp
        else:
            results['keypoint'] = skeleton

        return results


@PIPELINES.register_module()
class Kinetics_Transform:
    """  coco_17 -> coco_20  """
    
    def __init__(self, dataset='coco_new'):
        self.dataset = dataset

    def __call__(self, results):
        
        if 'keypoint_score' in results and 'keypoint' in results:
            assert self.dataset != 'nturgb+d'
            assert results['keypoint'].shape[-1] == 2, 'Only 2D keypoints have keypoint_score. '
            keypoint = results.pop('keypoint')
            keypoint_score = results.pop('keypoint_score')
            results['keypoint'] = np.concatenate([keypoint, keypoint_score[..., None]], -1)
        
        # M T V C
        skeleton = results['keypoint']
        M, T, V, C = skeleton.shape
        # M T V C -> V M T C
        skeleton = skeleton.transpose(2, 0, 1, 3)
        add_term = np.zeros((3, M, T, C))
        
        add_term[0] = (skeleton[11] + skeleton[12]) / 2
        add_term[2] = (skeleton[5] + skeleton[6]) / 2
        add_term[1] = (add_term[0] + add_term[2]) / 2
        
        skeleton = np.concatenate([skeleton, add_term], 0)
        # V M T C -> M T V C
        skeleton = skeleton.transpose(1, 2, 0, 3).astype(np.float32)
        results['keypoint'] = skeleton

        return results


@PIPELINES.register_module()
class JointToBone:

    def __init__(self, dataset='nturgb+d', target='keypoint'):
        self.dataset = dataset
        self.target = target
        if self.dataset not in ['nturgb+d', 'openpose', 'openpose_new', 'coco', 'coco_new', 'custom_17', 'openpose_body25']:
            raise ValueError(
                f'The dataset type {self.dataset} is not supported')
        if self.dataset == 'nturgb+d':
            self.pairs = ((0, 1), (1, 20), (2, 20), (3, 2), (4, 20), (5, 4), (6, 5), (7, 6), (8, 20), (9, 8),
                          (10, 9), (11, 10), (12, 0), (13, 12), (14, 13), (15, 14), (16, 0), (17, 16), (18, 17),
                          (19, 18), (21, 22), (20, 20), (22, 7), (23, 24), (24, 11))
        elif self.dataset == 'openpose':
            self.pairs = ((0, 1), (1, 1), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6), (8, 2), (9, 8), (10, 9),
                          (11, 5), (12, 11), (13, 12), (14, 0), (15, 0), (16, 14), (17, 15))
        elif self.dataset == 'openpose_new':
            self.pairs = ((0, 1), (1, 1), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5), (7, 6), (8, 18), (9, 8), (10, 9),
                          (11, 18), (12, 11), (13, 12), (14, 0), (15, 0), (16, 14), (17, 15), (18, 19), (19, 1))
        elif self.dataset == 'coco':
            self.pairs = ((0, 0), (1, 0), (2, 0), (3, 1), (4, 2), (5, 0), (6, 0), (7, 5), (8, 6), (9, 7), (10, 8),
                          (11, 0), (12, 0), (13, 11), (14, 12), (15, 13), (16, 14))
        elif self.dataset == 'coco_new':
            self.pairs = ((0, 19), (1, 0), (2, 0), (3, 1), (4, 2), (5, 19), (6, 19), (7, 5), (8, 6), (9, 7), (10, 8),
                          (11, 17), (12, 17), (13, 11), (14, 12), (15, 13), (16, 14), (17, 18), (18, 19), (19, 19))
        elif self.dataset == 'custom_17':
            # Graph의 inward와 거의 동일하지만, 중심점(0, 8)에 대한 셀프 링크를 추가하고,
            # 물리적 연결을 그대로 따르는 것이 가장 안정적이라 판단하여 동일하게 구성합니다.
            self.pairs = (
                (0, 0),  # 중심점 0에 대한 셀프 링크
                (1, 0), (2, 1), (3, 2),
                (4, 0), (5, 4), (6, 5),
                (7, 0),
                (8, 8),  # 상체 중심점 8에 대한 셀프 링크
                (9, 8), (10, 9),
                (11, 8), (12, 11), (13, 12),
                (14, 8), (15, 14), (16, 15)
            )

        elif self.dataset == 'openpose_body25':
            self.pairs = (
                (0, 1), (1, 1), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5),
                (7, 6), (8, 1), (9, 8), (10, 9), (11, 10), (12, 8),
                (13, 12), (14, 13), (15, 0), (16, 0), (17, 15), (18, 16),
                (19, 14), (20, 19), (21, 14), (22, 11), (23, 22), (24, 11)
            )

    def __call__(self, results):

        keypoint = results['keypoint']
        M, T, V, C = keypoint.shape
        bone = np.zeros((M, T, V, C), dtype=np.float32)

        assert C in [2, 3]
        for v1, v2 in self.pairs:
            bone[..., v1, :] = keypoint[..., v1, :] - keypoint[..., v2, :]
            if C == 3 and self.dataset in ['openpose', 'openpose_new', 'coco', 'coco_new', 'handmp', 'openpose_body25', 'custom_17']:
                score = (keypoint[..., v1, 2] + keypoint[..., v2, 2]) / 2
                bone[..., v1, 2] = score

        results[self.target] = bone
        return results


@PIPELINES.register_module()
class JointToKB:

    def __init__(self, dataset='nturgb+d', target='keypoint'):
        self.dataset = dataset
        self.target = target
        if self.dataset not in ['nturgb+d', 'openpose', 'openpose_new', 'coco', 'coco_new', 'openpose_body25', 'custom_17']:
            raise ValueError(
                f'The dataset type {self.dataset} is not supported')
        if self.dataset == 'nturgb+d':
            self.pairs = ((0, 20), (1, 1), (2, 2), (3, 20), (4, 4), (5, 20), (6, 4), (7, 5), (8, 8), (9, 20),
                          (10, 8), (11, 9), (12, 1), (13, 0), (14, 12), (15, 13), (16, 1), (17, 0), (18, 16),
                          (19, 17), (21, 7), (20, 20), (22, 6), (23, 11), (24, 10))
        elif self.dataset == 'openpose':
            self.pairs = ((0, 0), (1, 1), (2, 2), (3, 1), (4, 2), (5, 5), (6, 1), (7, 5), (8, 1), (9, 2), (10, 8),
                          (11, 1), (12, 5), (13, 11), (14, 1), (15, 1), (16, 0), (17, 0))
        elif self.dataset == 'openpose_new':
            self.pairs = ((0, 0), (1, 1), (2, 2), (3, 1), (4, 2), (5, 5), (6, 1), (7, 5), (8, 19), (9, 18), (10, 8),
                          (11, 19), (12, 18), (13, 11), (14, 1), (15, 1), (16, 0), (17, 0), (18, 1), (19, 19))
        elif self.dataset == 'coco':
            self.pairs = ((0, 0), (1, 1), (2, 2), (3, 0), (4, 0), (5, 5), (6, 6), (7, 0), (8, 0), (9, 5), (10, 6),
                          (11, 11), (12, 12), (13, 0), (14, 0), (15, 11), (16, 12))
        elif self.dataset == 'coco_new':
            self.pairs = ((0, 0), (1, 19), (2, 19), (3, 0), (4, 0), (5, 5), (6, 6), (7, 19), (8, 19), (9, 5), 
                          (10, 6),(11, 18), (12, 18), (13, 17), (14, 17), (15, 11), (16, 12), (17, 19), (18, 18), (19, 19))
        # 💡 [수정 5] custom_17에 대한 pairs 규칙 추가 (JointToBone과 동일하게)
        elif self.dataset == 'custom_17':
            self.pairs = (
                (0, 0), (1, 0), (2, 1), (3, 2), (4, 0), (5, 4),
                (6, 5), (7, 0), (8, 8), (9, 8), (10, 9),
                (11, 8), (12, 11), (13, 12), (14, 8), (15, 14), (16, 15)
            )

        # 💡 [수정 6] openpose_body25에 대한 pairs 규칙 추가 (JointToBone과 동일하게)
        elif self.dataset == 'openpose_body25':
            self.pairs = (
                (0, 1), (1, 1), (2, 1), (3, 2), (4, 3), (5, 1), (6, 5),
                (7, 6), (8, 1), (9, 8), (10, 9), (11, 10), (12, 8),
                (13, 12), (14, 13), (15, 0), (16, 0), (17, 15), (18, 16),
                (19, 14), (20, 19), (21, 14), (22, 11), (23, 22), (24, 11)
            )

    def __call__(self, results):

        keypoint = results['keypoint']
        M, T, V, C = keypoint.shape
        bone = np.zeros((M, T, V, C), dtype=np.float32)

        assert C in [2, 3]
        for v1, v2 in self.pairs:
            bone[..., v1, :] = keypoint[..., v1, :] - keypoint[..., v2, :]
            if C == 3 and self.dataset in ['openpose', 'coco','custom_17','openpose_body25']:
                score = (keypoint[..., v1, 2] + keypoint[..., v2, 2]) / 2
                bone[..., v1, 2] = score

        results[self.target] = bone
        return results


@PIPELINES.register_module()
class ToMotion:

    def __init__(self, dataset='nturgb+d', source='keypoint', target='motion'):
        self.dataset = dataset
        self.source = source
        self.target = target

    def __call__(self, results):
        data = results[self.source]
        M, T, V, C = data.shape
        motion = np.zeros_like(data)

        assert C in [2, 3]
        motion[:, :T - 1] = np.diff(data, axis=1)
        if C == 3 and self.dataset in ['openpose', 'coco','custom_17','openpose_body25']:
            score = (data[:, :T - 1, :, 2] + data[:, 1:, :, 2]) / 2
            motion[:, :T - 1, :, 2] = score

        results[self.target] = motion

        return results


@PIPELINES.register_module()
class MergeSkeFeat:
    def __init__(self, feat_list=['keypoint'], target='keypoint', axis=-1):
        """Merge different feats (ndarray) by concatenate them in the last axis. """

        self.feat_list = feat_list
        self.target = target
        self.axis = axis

    def __call__(self, results):
        feats = []
        for name in self.feat_list:
            feats.append(results.pop(name))
        feats = np.concatenate(feats, axis=self.axis)
        results[self.target] = feats
        return results


@PIPELINES.register_module()
class GenSkeFeat:
    def __init__(self, dataset='nturgb+d', feats=['j'], axis=-1):
        self.dataset = dataset
        self.feats = feats
        self.axis = axis
        ops = []
        if 'b' in feats or 'bm' in feats:
            ops.append(JointToBone(dataset=dataset, target='b'))
        if 'k' in feats or 'km' in feats:
            ops.append(JointToKB(dataset=dataset, target='k'))
        ops.append(Rename({'keypoint': 'j'}))
        if 'jm' in feats:
            ops.append(ToMotion(dataset=dataset, source='j', target='jm'))
        if 'bm' in feats:
            ops.append(ToMotion(dataset=dataset, source='b', target='bm'))
        if 'km' in feats:
            ops.append(ToMotion(dataset=dataset, source='k', target='km'))
        ops.append(MergeSkeFeat(feat_list=feats, axis=axis))
        self.ops = Compose(ops)

    def __call__(self, results):
        if 'keypoint_score' in results and 'keypoint' in results:
            assert self.dataset != 'nturgb+d'
            assert results['keypoint'].shape[-1] == 2, 'Only 2D keypoints have keypoint_score. '
            keypoint = results.pop('keypoint')
            keypoint_score = results.pop('keypoint_score')
            results['keypoint'] = np.concatenate([keypoint, keypoint_score[..., None]], -1)
        return self.ops(results)


@PIPELINES.register_module()
class FormatGCNInput:
    """Format final skeleton shape to the given input_format. """

    def __init__(self, num_person=2, mode='zero'):
        self.num_person = num_person
        assert mode in ['zero', 'loop']
        self.mode = mode

    def __call__(self, results):
        """Performs the FormatShape formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        keypoint = results['keypoint']
        if 'keypoint_score' in results:
            keypoint = np.concatenate((keypoint, results['keypoint_score'][..., None]), axis=-1)

        # M T V C
        if keypoint.shape[0] < self.num_person:
            pad_dim = self.num_person - keypoint.shape[0]
            pad = np.zeros((pad_dim, ) + keypoint.shape[1:], dtype=keypoint.dtype)
            keypoint = np.concatenate((keypoint, pad), axis=0)
            if self.mode == 'loop' and keypoint.shape[0] == 1:
                for i in range(1, self.num_person):
                    keypoint[i] = keypoint[0]

        elif keypoint.shape[0] > self.num_person:
            keypoint = keypoint[:self.num_person]

        M, T, V, C = keypoint.shape
        nc = results.get('num_clips', 1)
        assert T % nc == 0
        keypoint = keypoint.reshape((M, nc, T // nc, V, C)).transpose(1, 0, 2, 3, 4)
        results['keypoint'] = np.ascontiguousarray(keypoint)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__ + f'(num_person={self.num_person}, mode={self.mode})'
        return repr_str


@PIPELINES.register_module()
class DecompressPose:
    """Load Compressed Pose

    In compressed pose annotations, each item contains the following keys:
    Original keys: 'label', 'frame_dir', 'img_shape', 'original_shape', 'total_frames'
    New keys: 'frame_inds', 'keypoint', 'anno_inds'.
    This operation: 'frame_inds', 'keypoint', 'total_frames', 'anno_inds'
         -> 'keypoint', 'keypoint_score', 'total_frames'

    Args:
        squeeze (bool): Whether to remove frames with no human pose. Default: True.
        max_person (int): The max number of persons in a frame, we keep skeletons with scores from high to low.
            Default: 10.
    """

    def __init__(self,
                 squeeze=True,
                 max_person=10):

        self.squeeze = squeeze
        self.max_person = max_person

    def __call__(self, results):

        required_keys = ['total_frames', 'frame_inds', 'keypoint']
        for k in required_keys:
            assert k in results

        total_frames = results['total_frames']
        frame_inds = results.pop('frame_inds')
        keypoint = results['keypoint']

        if 'anno_inds' in results:
            frame_inds = frame_inds[results['anno_inds']]
            keypoint = keypoint[results['anno_inds']]

        assert np.all(np.diff(frame_inds) >= 0), 'frame_inds should be monotonical increasing'

        def mapinds(inds):
            uni = np.unique(inds)
            map_ = {x: i for i, x in enumerate(uni)}
            inds = [map_[x] for x in inds]
            return np.array(inds, dtype=np.int16)

        if self.squeeze:
            frame_inds = mapinds(frame_inds)
            total_frames = np.max(frame_inds) + 1

        results['total_frames'] = total_frames

        num_joints = keypoint.shape[1]
        num_person = get_mode(frame_inds)[-1][0]

        new_kp = np.zeros([num_person, total_frames, num_joints, 2], dtype=np.float16)
        new_kpscore = np.zeros([num_person, total_frames, num_joints], dtype=np.float16)
        
        nperson_per_frame = np.zeros([total_frames], dtype=np.int16)

        for frame_ind, kp in zip(frame_inds, keypoint):
            person_ind = nperson_per_frame[frame_ind]
            new_kp[person_ind, frame_ind] = kp[:, :2]
            new_kpscore[person_ind, frame_ind] = kp[:, 2]
            nperson_per_frame[frame_ind] += 1

        if num_person > self.max_person:
            for i in range(total_frames):
                nperson = nperson_per_frame[i]
                val = new_kpscore[:nperson, i]
                score_sum = val.sum(-1)

                inds = sorted(range(nperson), key=lambda x: -score_sum[x])
                new_kpscore[:nperson, i] = new_kpscore[inds, i]
                new_kp[:nperson, i] = new_kp[inds, i]
            num_person = self.max_person
            results['num_person'] = num_person

        results['keypoint'] = new_kp[:num_person]
        results['keypoint_score'] = new_kpscore[:num_person]
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(squeeze={self.squeeze}, max_person={self.max_person})')
