# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.core.bbox.samplers.base_sampler import BaseSampler
from mmdet.core.bbox.samplers.sampling_result import SamplingResult

from ..builder import ROTATED_BBOX_SAMPLERS


@ROTATED_BBOX_SAMPLERS.register_module()
class RPseudoSampler(BaseSampler):
    """A pseudo sampler that does not do sampling actually."""

    def __init__(self, with_score=True, **kwargs):
        self.with_score = with_score

    def _sample_pos(self, **kwargs):
        """Sample positive samples."""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative samples."""
        raise NotImplementedError

    def sample(self, assign_result, bboxes, gt_bboxes, *args, **kwargs):
        """Directly returns the positive and negative indices  of samples.
        Args:
            assign_result (:obj:`AssignResult`): Assigned results
            bboxes (torch.Tensor): Bounding boxes
            gt_bboxes (torch.Tensor): Ground truth boxes
        Returns:
            :obj:`SamplingResult`: sampler results
        """
        if self.with_score:
            bboxes = bboxes[:, :6]
        else:
            bboxes = bboxes[:, :5]
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        gt_flags = bboxes.new_zeros(bboxes.shape[0], dtype=torch.uint8)
        sampling_result = SamplingResult(pos_inds, neg_inds, bboxes, gt_bboxes,
                                         assign_result, gt_flags)
        return sampling_result
