# 파킨슨 데이터셋을 위한 모든 modality 통합 설정

# 1. 사용할 모든 modality 정의
modalities = ['j', 'b', 'k', 'jm', 'bm', 'km']

# 2. modality 개수에 따라 입력 채널 수 계산
# 원본 설정에서 modality당 2개의 채널(예: x, y)을 사용했음
in_channels = 2 * len(modalities)

# 3. 제공된 설정값을 기반으로 통합 모델 파라미터 설정
_num_prototype = 100
_weight = 0.3

graph = 'openpose_body25'
# work_dir: 모델 가중치와 로그가 저장될 경로
work_dir = f'./work_dirs/custom/new_openpose_25__not_noise_kalman/parkinson_unified'

# 4. 새로 계산된 in_channels로 모델 정의
model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='ProtoGCN',
        in_channels=in_channels,  # 계산된 in_channels 전달
        num_prototype=_num_prototype,
        tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
        graph_cfg=dict(layout=graph, mode='random', num_filter=8, init_off=.04, init_std=.02)),
    # num_classes: '파킨슨병'과 '정상' 2개의 클래스로 설정
    cls_head=dict(type='SimpleHead', joint_cfg='openpose_body25', num_classes=2, in_channels=384, weight=_weight))

dataset_type = 'PoseDataset'
# ann_file: 데이터 정보가 담긴 .pkl 파일 경로
ann_file = '/NAS/dclab/psw/agi/ProtoGCN/data/custom/not_noise_pkl/not_noise_pkl_kalman.pkl'

# openpose_body25 레이아웃에 맞는 좌우 관절 인덱스
left_kp = [5, 6, 7, 12, 13, 14, 16, 18, 19, 20, 21]
right_kp = [2, 3, 4, 9, 10, 11, 15, 17, 22, 23, 24]

# 5. 모든 modality 리스트를 사용하도록 파이프라인 업데이트
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=100),
    dict(type='PoseDecode'),
    # dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp), # 원본 파일과 동일하게 Flip 증강은 주석 처리됨
    dict(type='GenSkeFeat', dataset='openpose_body25', feats=modalities),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=100, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='GenSkeFeat', dataset='openpose_body25', feats=modalities),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=100, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='GenSkeFeat', dataset='openpose_body25', feats=modalities),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

data = dict(
    videos_per_gpu=16,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    # .pkl 파일에 저장된 split 이름과 일치시킴
    train=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='train'),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='test'))

# Optimizer, lr_config 등은 그대로 유지
optimizer = dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 150 
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
gpu_ids = range(0, 1)

# 재현성을 위한 설정
seed = 42
deterministic = True
