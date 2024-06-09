# Copyright (c) OpenMMLab. All rights reserved.
from .convfc_rbbox_head import (RotatedConvFCBBoxHead,
                                RotatedKFIoUShared2FCBBoxHead,
                                RotatedShared2FCBBoxHead,
                                RotatedShared4Conv1FCBBoxHead)
from .double_rbbox_head import RotatedDoubleConvFCBBoxHead
from .gv_bbox_head import GVBBoxHead
from .rotated_bbox_head import RotatedBBoxHead
from .convfc_rbbox_arl_head import RotatedShared2FCBBoxARLHead, RotatedShared4Conv1FCBBoxARLHead
from .orth_convfc_rbbox_arl_head import ORTH_RotatedConvFCBBoxARLHead, ORTH_RotatedShared2FCBBoxARLHead, ORTH_RotatedShared4Conv1FCBBoxARLHead
from .orth_convfc_rbbox_head import (ORTH_RotatedConvFCBBoxHead,
                                ORTH_RotatedKFIoUShared2FCBBoxHead,
                                ORTH_RotatedShared2FCBBoxHead,
                                ORTH_RotatedShared4Conv1FCBBoxHead)

__all__ = [
    'RotatedBBoxHead', 'RotatedConvFCBBoxHead', 'RotatedShared2FCBBoxHead', 'RotatedShared4Conv1FCBBoxHead',
    'GVBBoxHead', 'RotatedKFIoUShared2FCBBoxHead', 'RotatedDoubleConvFCBBoxHead', 'RotatedShared2FCBBoxARLHead',
    'RotatedShared4Conv1FCBBoxARLHead',
    'ORTH_RotatedConvFCBBoxARLHead', 'ORTH_RotatedShared2FCBBoxARLHead', 'ORTH_RotatedShared4Conv1FCBBoxARLHead',
    'ORTH_RotatedConvFCBBoxHead', 'ORTH_RotatedKFIoUShared2FCBBoxHead', 'ORTH_RotatedShared2FCBBoxHead', 'ORTH_RotatedShared4Conv1FCBBoxHead'
]
