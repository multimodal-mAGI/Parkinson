modality = 'bm'
graph = 'mediapipe'
# work_dir: 모델 가중치와 로그가 저장될 경로입니다.
work_dir = f'./work_dirs/custom/openpose_25/parkinson_bm'
load_from = '/NAS/dclab/psw/agi/ProtoGCN/checkpoint/best_top1_acc_epoch_150_b.pth'
model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='ProtoGCN',
        # 3D 좌표(x, y, confidence)를 사용하므로 in_channels를 3으로 설정
        in_channels=2,
        num_prototype=100,
        tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
        graph_cfg=dict(layout=graph, mode='random', num_filter=8, init_off=.04, init_std=.02)),
    # num_classes: 분류할 클래스의 수입니다. '파킨슨병'과 '정상' 2개로 설정
    cls_head=dict(type='SimpleHead', joint_cfg='mediapipe', num_classes=2, in_channels=384, weight=0.3))

dataset_type = 'PoseDataset'
# ann_file: 데이터 정보가 담긴 .pkl 파일의 경로입니다.
ann_file = '/NAS/dclab/psw/agi/ProtoGCN/data/custom/parkinson_data_good.pkl'

# 'mediapipe' 25-node 그래프 레이아웃에 맞춘 좌우 관절 인덱스
# (Left Arm: 8, 9, 10, 11, 23, 24) + (Left Leg: 16, 17, 18, 19)
left_kp = [8, 9, 10, 11, 23, 24, 16, 17, 18, 19]
# (Right Arm: 4, 5, 6, 7, 21, 22) + (Right Leg: 12, 13, 14, 15)
right_kp = [4, 5, 6, 7, 21, 22, 12, 13, 14, 15]

train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=100),
    dict(type='PoseDecode'),
    # 데이터 증강을 위해 Flip 파이프라인 추가
    # dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GenSkeFeat', dataset='mediapipe', feats=[modality]),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=100, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='GenSkeFeat', dataset='mediapipe', feats=[modality]),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=100, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='GenSkeFeat', dataset='mediapipe', feats=[modality]),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    # .pkl 파일에 저장된 split 이름('xsub_train', 'xsub_val')과 일치시킵니다.
    train=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='train'),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='test'))

# optimizer, lr_config 등 나머지 설정은 그대로 유지
optimizer = dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 300 
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
gpu_ids = range(0, 1)