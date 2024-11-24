"""
******************************
Notes:
The current BRL repo uses the older version of the PYSKL code.
However, the K400 config needs to be run under the latest version of PYSKL.
You could refer to my new repo ProtoGCN to adapt it for a better performance.
code: https://github.com/firework8/ProtoGCN
******************************
"""
modality = 'jm'
graph = 'coco'
work_dir = f'./work_dirs/strong_aug_k400_hrnet/{modality}'

model = dict(
    type='MixRecognizerGCN',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        gcn_with_res=True,
        tcn_type='mstcn',
        graph_cfg=dict(layout=graph, mode='spatial')),
    cls_head=dict(type='MixKAAHead', num_classes=400, in_channels=256))

memcached = True
mc_cfg = ('localhost', 22077)
dataset_type = 'PoseDataset'
ann_file = '/data/lhd/pyskl_data/k400/k400_hrnet.pkl'

left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
box_thr = 0.5
valid_ratio = 0.0

train_pipeline = [
    dict(type='DecompressPose', squeeze=True),
    dict(type='UniformSampleFrames', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GenSkeFeat', dataset='coco', feats=[modality]),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='DecompressPose', squeeze=True),
    dict(type='UniformSampleFrames', clip_len=100, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='GenSkeFeat', dataset='coco', feats=[modality]),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='DecompressPose', squeeze=True),
    dict(type='UniformSampleFrames', clip_len=100, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='GenSkeFeat', dataset='coco', feats=[modality]),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=64,
    workers_per_gpu=16,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type=dataset_type, ann_file=ann_file, split='train', pipeline=train_pipeline,
        box_thr=box_thr, valid_ratio=valid_ratio, memcached=memcached, mc_cfg=mc_cfg),
    val=dict(
        type=dataset_type, ann_file=ann_file, split='val', pipeline=val_pipeline,
        box_thr=box_thr, memcached=memcached, mc_cfg=mc_cfg),
    test=dict(
        type=dataset_type, ann_file=ann_file, split='val', pipeline=test_pipeline,
        box_thr=box_thr, memcached=memcached, mc_cfg=mc_cfg))

# optimizer, 4GPU lr=0.1
# optimizer 128@0.1; 16-> 0.0125 ; 32-> 0.025 ; 64-> 0.05
# optimizer 128@0.2; 16-> 0.025 ; 32-> 0.05 ; 64-> 0.1
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 150
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], topk=(1, 5))
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
