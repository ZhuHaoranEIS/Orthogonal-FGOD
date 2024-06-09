_base_ = ['./qopn_rcnn_bcfn_r50_fpn_100e_shiprs3_le90.py']

model = dict(
    rpn_head=dict(
        loss_cls=dict(
            loss_weight=0.5
        ),
        loss_bbox=dict(
            loss_weight=0.5
        )
    ),
    roi_head=dict(
        bbox_head=dict(
            type='RotatedShared2FCBBoxARLHead',
            loss_cls=dict(
                type='AdaptiveRecognitionLoss',
                beta=2.5,
                gamma=1.5),
        )
    ),
    train_cfg=dict(
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=None,
        ),
        rcnn=dict(
            sampler=dict(
                _delete_=True,
                type='RPseudoSampler')),
    ),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=None),
        rcnn=dict(
            nms_pre=1000,
            max_per_img=1000)
    )
)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=1.0 / 2000,
    step=[24, 33])
