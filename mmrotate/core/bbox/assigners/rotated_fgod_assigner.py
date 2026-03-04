# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import BBOX_ASSIGNERS
from ..iou_calculators.builder import build_iou_calculator
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner


@BBOX_ASSIGNERS.register_module()
class RotatedFGODAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `0` or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        topk (float): Number of bbox selected in each level.
    """

    def __init__(self,
                 topk,
                 max_score=3.0,
                 iou_calculator=dict(type='RBboxOverlaps2D'),
                 ignore_iof_thr=-1):
        self.topk = topk
        self.max_score = max_score
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.ignore_iof_thr = ignore_iof_thr

    def assign(self,
               bboxes,
               num_level_bboxes,
               gt_bboxes,
               gt_bboxes_ignore=None,
               gt_labels=None,
               cls_scores=None):
        """Assign gt to bboxes.

        The assignment is done in following steps

        1. compute iou between all bbox (bbox of all pyramid levels) and gt
        2. compute center distance between all bbox and gt
        3. on each pyramid level, for each gt, select k bbox whose center
           are closest to the gt center, so we total select k*l bbox as
           candidates for each gt
        4. get corresponding iou for the these candidates, and compute the
           mean and std, set mean + std as the iou threshold
        5. select these candidates whose iou are greater than or equal to
           the threshold as positive
        6. limit the positive sample's center in gt

        Args:
            bboxes (Tensor): Bounding boxes to be assigned, shape(n, 5).
            num_level_bboxes (List): num of bboxes in each level
            gt_bboxes (Tensor): Groundtruth boxes, shape (k, 5).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        INF = 100000000
        RATIO = 20000.0
        bboxes = bboxes[:, :5]
        num_gt, num_bboxes = gt_bboxes.size(0), bboxes.size(0)

        # compute iou between all bbox and gt
        assert gt_bboxes.size(1) == 5
        overlaps = self.iou_calculator(bboxes, gt_bboxes)

        # assign 0 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                             0,
                                             dtype=torch.long)
        
        gt_cx, gt_cy = gt_bboxes[:, 0], gt_bboxes[:, 1]
        gt_points = torch.stack((gt_cx, gt_cy), dim=1)

        bboxes_cx, bboxes_cy = bboxes[:, 0], bboxes[:, 1]
        bboxes_points = torch.stack((bboxes_cx, bboxes_cy), dim=1)
        
        distances = (bboxes_points[:, None, :] -
                     gt_points[None, :, :]).pow(2).sum(-1).sqrt()

        if num_gt == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes, ))
            if num_gt == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes, ),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and bboxes.numel() > 0):
            ignore_overlaps = self.iou_calculator(
                bboxes, gt_bboxes_ignore, mode='iof')
            ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            ignore_idxs = ignore_max_overlaps > self.ignore_iof_thr
            distances[ignore_idxs, :] = INF
            assigned_gt_inds[ignore_idxs] = -1

        selectable_k = min(overlaps.shape[0], self.topk)
        overlaps = overlaps*RATIO - distances # num_samples, num_gt
        candidate_idxs = overlaps.topk(selectable_k, dim=0, largest=True)[1] # topk, num_gt
        
        if cls_scores is not None:
            candidate_scores = cls_scores[candidate_idxs].max(dim=-1)[0] # topk, num_gt
            candidate_scores_sum = candidate_scores.sum(dim=0) # num_gt
            is_pos = self.filter_candidates_by_score_sum(candidate_scores, self.max_score)
            # print(candidate_scores_sum)
        else:
            # get corresponding iou for the these candidates, and compute the
            # mean and std, set mean + std as the iou threshold
            candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)]
            overlaps_mean_per_gt = candidate_overlaps.mean(0)
            overlaps_std_per_gt = candidate_overlaps.std(0)
            overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

            is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]
            

        # limit the positive sample's center in gt
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        # ep_bboxes_cx = bboxes_cx.view(1, -1).expand(
        #     num_gt, num_bboxes).contiguous().view(-1)
        # ep_bboxes_cy = bboxes_cy.view(1, -1).expand(
        #     num_gt, num_bboxes).contiguous().view(-1)
        candidate_idxs = candidate_idxs.view(-1)

        # # calculate the left, top, right, bottom distance between positive
        # # bbox center and gt side
        # gt_ctr, gt_wh, gt_thetas = torch.split(
        #     gt_bboxes, [2, 2, 1], dim=1)
        # ep_bboxes_cxcy = torch.stack([ep_bboxes_cx[candidate_idxs].view(-1, num_gt),
        #                               ep_bboxes_cy[candidate_idxs].view(-1, num_gt)], 2)
        # offset = ep_bboxes_cxcy - gt_ctr
        # cos, sin = torch.cos(gt_thetas), torch.sin(gt_thetas)
        # Matrix = torch.cat([cos, sin, -sin, cos], dim=-1).reshape(
        #     offset.shape[-2], 2, 2)
        # offset = torch.matmul(Matrix, offset[..., None])
        # offset = offset.squeeze(-1)
        # W, H = gt_wh[:, 0], gt_wh[:, 1]
        # offset_x, offset_y = offset[..., 0], offset[..., 1]
        # l_ = W / 2 + offset_x
        # r_ = W / 2 - offset_x
        # t_ = H / 2 + offset_y
        # b_ = H / 2 - offset_y
        # is_in_gts = torch.stack([l_, t_, r_, b_], dim=1).min(dim=1)[0] > 1e-8

        # is_pos = is_pos & is_in_gts

        # if an anchor box is assigned to multiple gts,
        # the one with the highest IoU will be selected.
        overlaps_inf = torch.full_like(overlaps,
                                       -INF).t().contiguous().view(-1)
        index = candidate_idxs.view(-1)[is_pos.view(-1)]
        overlaps_inf[index] = overlaps.t().contiguous().view(-1)[index]
        overlaps_inf = overlaps_inf.view(num_gt, -1).t()

        max_overlaps, argmax_overlaps = overlaps_inf.max(dim=1)
        assigned_gt_inds[
            max_overlaps != -INF] = argmax_overlaps[max_overlaps != -INF] + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None
        return AssignResult(
            num_gt, assigned_gt_inds, max_overlaps, labels=assigned_labels)


    def filter_candidates_by_score_sum(self, candidate_scores, max_scores):
        """
        Args:
            candidate_scores: (topk, num_gt) - 这里的顺序对应 overlaps.topk 选出来的顺序
            max_scores: float 或 (num_gt,) - 分数总和上限
            
        Returns:
            is_pos: (topk, num_gt) - 布尔类型，可以直接用于掩膜筛选
        """
        
        # 1. 对分数进行降序排列 (Top-K 维度)
        # 我们希望在删除时，保留分数高的，删除分数低的
        sorted_scores, sort_idxs = candidate_scores.sort(dim=0, descending=True)
        
        # 2. 计算累加得分 (Cumulative Sum)
        cum_scores = sorted_scores.cumsum(dim=0)
        
        # 3. 生成截断掩码 (在排序状态下)
        # 如果累加值小于等于 max_scores，则保留
        # 这里的逻辑是严格限制：一旦总和超过 max_scores，后面的全部丢弃
        keep_mask_sorted = cum_scores <= max_scores
        
        # [可选保护策略] 
        # 如果 top1 的分数本身就大于 max_scores，上述逻辑会导致该 GT 没有正样本。
        # 通常做法是强制保留第一个（得分最高的那个），防止梯度消失。
        # keep_mask_sorted[0, :] = True 
        
        # 4. 将掩码还原回原始顺序 (Scatter)
        # 初始化全 False 的 mask
        is_pos = torch.zeros_like(candidate_scores, dtype=torch.bool)
        
        # 使用 scatter_ 将排序后的结果映射回原始索引位置
        # dim=0, index=sort_idxs, src=keep_mask_sorted
        is_pos.scatter_(0, sort_idxs, keep_mask_sorted)
        
        return is_pos