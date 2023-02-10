model = dict(
    type='RecognizerGCN',
    backbone=dict(
		type='MSG3D',
        graph_cfg=dict(layout='nturgb+d', mode='binary_adj')),
    cls_head=dict(type='GCNHead', num_classes=60, in_channels=384))

dataset_type = 'PoseDataset'
ann_file = '/data/lhd/pyskl_data/nturgbd/ntu60_3danno.pkl'
imb_factor = '100'

sampel_class_prob = None

train_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['bm']),
    dict(type='UniformSample', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['bm']),
    dict(type='UniformSample', clip_len=100, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['bm']),
    dict(type='UniformSample', clip_len=100, num_clips=10, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=2,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(
        type='RepeatDataset',
        times=5,
        num_classes=60,
        data_list_path='./data/NTU60_LT/NTU60_xview_exp_'+imb_factor+'.txt',
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, 
        class_prob=sampel_class_prob, split='xview_train')),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='xview_val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='xview_val'))

# optimizer
optimizer = dict(type='SGD', lr=0.0125, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 16
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])

# runtime settings
log_level = 'INFO'
work_dir = './work_dirs/msg3d/ntu60_xview_LT_msg3d/exp_' + imb_factor + '/bm'
