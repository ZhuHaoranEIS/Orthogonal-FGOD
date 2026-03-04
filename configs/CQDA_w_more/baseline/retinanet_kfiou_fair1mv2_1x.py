'''
CUDA_VISIBLE_DEVICES=3 nohup python tools/train.py \
    /home/zhr/PETDet/configs/Remote_sensing_exp/FAIR1Mv2/reproduction/retinanet_kfiou_fair1mv2_1x.py \
        --work-dir /data/zhr/work_dirs_file/Remote_sensing_FGOD/retinanet_kfiou_fair1mv2_1x \
            >> /home/zhr/PETDet/configs/Remote_sensing_exp/FAIR1Mv2/reproduction/log/retinanet_kfiou_fair1mv2_1x.out &
'''

_base_ = [
    '../../../_base_/schedules/schedule_1x.py',
    '../../../_base_/default_runtime.py'
]
angle_version = 'le90'

# model settings
model = dict(
    type='RotatedRetinaNet',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5),
    bbox_head=dict(
        type='KFIoURRetinaHead',
        num_classes=37,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        assign_by_circumhbbox=None,
        anchor_generator=dict(
            type='RotatedAnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[1.0, 0.5, 2.0],
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHAOBBoxCoder',
            angle_range=angle_version,
            norm_factor=None,
            edge_swap=True,
            proj_xy=True,
            target_means=(.0, .0, .0, .0, .0),
            target_stds=(1.0, 1.0, 1.0, 1.0, 1.0)),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='KFLoss', loss_weight=5.0)),
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
            iou_calculator=dict(type='RBboxOverlaps2D')),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=2000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(iou_thr=0.1),
        max_per_img=2000))

# dataset settings
dataset_type = 'FAIR1MDataset'
train_root = '/data/zhr/FAIR1M/fair1m/submission/v1/train/crop/'
# train_root = '/data/zhr/FAIR1M/fair1m/submission/trainval/'
test_root = '/data/zhr/FAIR1M/fair1m/submission/v2/validation/crop/'
# test_root = '/data/zhr/FAIR1M/fair1m/submission/v2/test/crop/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=train_root + 'annfiles/',
        img_prefix=train_root + 'images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=test_root + 'annfiles/',
        img_prefix=test_root + 'images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=test_root + 'annfiles/',
        img_prefix=test_root + 'images/',
        pipeline=test_pipeline))

# optimizer
# optimizer = dict(_delete_=True, type='AdamW', lr=0.000025, weight_decay=0.0001)
# optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=1, norm_type=2))
optimizer = dict(lr=0.01)
# learning policy
lr_config = dict(policy='step', step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

evaluation = dict(interval=12, metric='mAP')
checkpoint_config = dict(interval=3)