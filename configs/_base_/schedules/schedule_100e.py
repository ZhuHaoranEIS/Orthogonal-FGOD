# evaluation
evaluation = dict(interval=50, metric='mAP')
# optimizer
optimizer = dict(type='SGD', lr=0.01/2, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=2000,
    warmup_ratio=1.0 / 2000,
    step=[66, 90])
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=100)
