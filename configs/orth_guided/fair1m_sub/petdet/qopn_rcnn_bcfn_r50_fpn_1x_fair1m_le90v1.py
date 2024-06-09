_base_ = ['/home/zhr/PETDet/configs/orth_guided/fair1m_sub/petdet/qopn_rcnn_r50_fpn_1x_fair1m_le90v1.py']

model = dict(
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=6),
    rpn_head=dict(
        start_level=1),
    fusion=dict(
        type='BCFN',
        feat_channels=256,
    ),
    roi_head=dict(
        bbox_roi_extractor=dict(
            featmap_strides=[4, 8, 16, 32],
        )
    )
)