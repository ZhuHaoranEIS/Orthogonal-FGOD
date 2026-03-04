import torch
import json
import numpy

from ..builder import build_bbox_coder
from mmdet.core.bbox.iou_calculators import build_iou_calculator
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner
from ..builder import ROTATED_BBOX_ASSIGNERS
#from mmcv.utils import build_from_cfg

from mmcv.ops import points_in_polygons
from ..transforms import obb2poly

@ROTATED_BBOX_ASSIGNERS.register_module()
class CAAssigner_v41(BaseAssigner):
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, or a semi-positive integer
    indicating the ground truth index.

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
        match_low_quality (bool): Whether to allow low quality matches. This is
            usually allowed for RPN and single stage detectors, but not allowed
            in the second stage. Details are demonstrated in Step 4.
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will assign
            on CPU device. Negative values mean not assign on CPU.
    """

    def __init__(self,
                 angle_version='le135',
                 gt_max_assign_all=True,
                 ignore_iof_thr=-1,
                 ignore_wrt_candidates=True,
                 gpu_assign_thr=512,
                 iou_calculator=dict(type='BboxOverlaps2D'),
                 assign_metric='gjsd',
                 topk=1,
                 topq=1,
                 constraint=False,
                 gauss_thr = 1.0,
                 bbox_coder=dict(
                     type='DeltaXYWHAOBBoxCoder',
                     target_means=(.0, .0, .0, .0, .0),
                     target_stds=(1.0, 1.0, 1.0, 1.0, 1.0))):
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.iou_calculator = build_iou_calculator(iou_calculator)
        self.assign_metric = assign_metric
        self.topk = topk
        self.topq = topq
        self.constraint = constraint
        self.gauss_thr = gauss_thr
        self.bbox_coder = build_bbox_coder(bbox_coder)
        
        self.angle_version = angle_version

    def assign(self, cls_scores, bbox_preds, bboxes, gt_bboxes, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to bboxes.
        
        The assignment is done in following steps

        1. compute gjsd between all bbox (bbox of all pyramid levels) and gt
        2. on each pyramid level, for each gt, select k bbox whose gjsd
            are largest to the gt center, so we total select k*l bbox as
            candidates for each gt
        3. get corresponding predicted quality[iou, cls] for the these candidates, and compute the
            mean and std, set mean + std as the quality threshold
        4. select these candidates whose quality are greater than or equal to
            the threshold as positive
        5. limit the positive sample's center with dgmm

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
        num_gt, num_bboxes = gt_bboxes.shape[0], bboxes.shape[0]
        box_dim = gt_bboxes.size(-1)

        # compute iou between all bbox and gt
        overlaps = self.iou_calculator(bboxes, gt_bboxes)

        # assign 0 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes, ),
                                            0,
                                            dtype=torch.long)
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

        # compute center distance between all bbox and gt
        # the center of gt and bbox
        gt_points = gt_bboxes[:, :2]
        bboxes_points = bboxes[:, :2]
        distances = (bboxes_points[:, None, :] -
                    gt_points[None, :, :]).pow(2).sum(-1).sqrt()

        # Selecting candidates based on the center distance
        candidate_idxs = []
        start_idx = 0

        print('num_bboxes: ', num_bboxes)
        if num_bboxes == 21824:
            num_level_bboxes = [16384, 4096, 1024, 256, 64]
        elif num_bboxes == 13343:
            num_level_bboxes = [10000, 2500, 625, 169, 49]
        elif num_bboxes == 10140:
            num_level_bboxes = [7600, 1900, 475, 130, 35]
        elif num_bboxes == 11210:
            num_level_bboxes = [8400, 2100, 525, 143, 42]
        else:
            num_level_bboxes = [10000, 2500, 625, 169, 49]
        assert sum(num_level_bboxes) == num_bboxes
        
        for level, bboxes_per_level in enumerate(num_level_bboxes):
            # on each pyramid level, for each gt,
            # select k bbox whose center are closest to the gt center
            end_idx = start_idx + bboxes_per_level
            distances_per_level = distances[start_idx:end_idx, :]
            _, topk_idxs_per_level = distances_per_level.topk(
                self.topk, dim=0, largest=False)
            candidate_idxs.append(topk_idxs_per_level + start_idx)
            start_idx = end_idx
        candidate_idxs = torch.cat(candidate_idxs, dim=0)

        # get corresponding iou for the these candidates, and compute the
        # mean and std, set mean + std as the iou threshold
        # gt_bboxes = obb2poly(gt_bboxes, self.angle_version)
        candidate_overlaps = overlaps[candidate_idxs, torch.arange(num_gt)]
        overlaps_mean_per_gt = candidate_overlaps.mean(0)
        overlaps_std_per_gt = candidate_overlaps.std(0)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt
        is_pos = candidate_overlaps >= overlaps_thr_per_gt[None, :]
        
        ###########################################################
        #### TODO: get the mean of the positive samples
        device = bboxes.device
        pos_prior_mean = torch.zeros((num_gt, 2), device=device)
        for gt_idx in range(num_gt):
            can_bbox_idxs_per_gt = candidate_idxs[:, gt_idx][is_pos[:, gt_idx]]
            # print('bboxes: ', bboxes.size())
            # print('can_bbox_idxs_per_gt: ', can_bbox_idxs_per_gt.size())
            # print('can_bbox_idxs_per_gt: ', can_bbox_idxs_per_gt)
            pos_prior_mean[gt_idx, :] = torch.mean(bboxes[can_bbox_idxs_per_gt, :2], dim=0)
        ###########################################################
        
        for gt_idx in range(num_gt):
            candidate_idxs[:, gt_idx] += gt_idx * num_bboxes
        candidate_idxs = candidate_idxs.view(-1)

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
        
        ### TODO: step 6: limit the positive sample's center with dgmm
        if self.constraint == 'dgmm':
            device1 = gt_bboxes.device
            if box_dim == 5:
                xy_gt, sigma_t = self.xy_wh_r_2_xy_sigma(gt_bboxes)
            elif box_dim == 4:
                xy_gt, sigma_t = self.xy_wh_2_xy_sigma(gt_bboxes)
            # get the mean of the positive samples
            # pos_prior_mean = torch.mean(assigned_pos_prior[...,:2], dim=-2)
            if box_dim == 5:
                _, sigma_t = self.xy_wh_r_2_xy_sigma(gt_bboxes)
            elif box_dim == 4:
                _, sigma_t = self.xy_wh_2_xy_sigma(gt_bboxes)
            xy_pt = pos_prior_mean
            xy_a = bboxes[...,:2]
            xy_gt = xy_gt[...,None,:,:2].unsqueeze(-1)
            xy_pt = xy_pt[...,None,:,:2].unsqueeze(-1)
            xy_a = xy_a[...,:,None,:2].unsqueeze(-1)
            inv_sigma_t = torch.stack((sigma_t[..., 1, 1], -sigma_t[..., 0, 1],
                                        -sigma_t[..., 1, 0], sigma_t[..., 0, 0]),
                                        dim=-1).reshape(-1, 2, 2)
            ###################################################################################################
            inv_sigma_t = inv_sigma_t / sigma_t.det().unsqueeze(-1).unsqueeze(-1)
            # inv_sigma_t = inv_sigma_t / sigma_t.cpu().det().cuda().unsqueeze(-1).unsqueeze(-1)
            # sigma_t_det = sigma_t[:,0,0]*sigma_t[:,1,1] - sigma_t[:,0,1]*sigma_t[:,1,0]
            # inv_sigma_t = inv_sigma_t / sigma_t_det.unsqueeze(-1).unsqueeze(-1)
            ###################################################################################################
            gaussian_gt = torch.exp(-0.5*(xy_a-xy_gt).permute(0, 1, 3, 2).matmul(inv_sigma_t).matmul(xy_a-xy_gt)).squeeze(-1).squeeze(-1)
            gaussian_pt = torch.exp(-0.5*(xy_a-xy_pt).permute(0, 1, 3, 2).matmul(inv_sigma_t).matmul(xy_a-xy_pt)).squeeze(-1).squeeze(-1)
            gaussian = 0.7*gaussian_gt + 0.3*gaussian_pt 
            inside_flag = gaussian >= torch.exp(torch.tensor([-self.gauss_thr])).to(device1)
            length = range(assigned_gt_inds.size(0))
            inside_mask = inside_flag[length, (assigned_gt_inds-1).clamp(min=0)]
            assigned_gt_inds *= inside_mask
        
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

    def assign_wrt_ranking(self,  overlaps, gt_labels=None):
        num_gts, num_bboxes = overlaps.size(0), overlaps.size(1)

        # 1. assign -1 by default
        assigned_gt_inds = overlaps.new_full((num_bboxes,),
                                             -1,
                                             dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            max_overlaps = overlaps.new_zeros((num_bboxes,))
            if num_gts == 0:
                # No truth, assign everything to background
                assigned_gt_inds[:] = 0
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = overlaps.new_full((num_bboxes,),
                                                    -1,
                                                    dtype=torch.long)
            return AssignResult(
                num_gts,
                assigned_gt_inds,
                max_overlaps,
                labels=assigned_labels)

        # for each anchor, which gt best overlaps with it
        # for each anchor, the max iou of all gts
        max_overlaps, _ = overlaps.max(dim=0)
        # for each gt, topk anchors
        # for each gt, the topk of all proposals
        gt_max_overlaps, _ = overlaps.topk(self.topk, dim=1, largest=True, sorted=True)  # gt_argmax_overlaps [num_gt, k]


        assigned_gt_inds[(max_overlaps >= 0)
                             & (max_overlaps < 0.8)] = 0

        for i in range(num_gts):
            for j in range(self.topk):
                max_overlap_inds = overlaps[i,:] == gt_max_overlaps[i,j]
                assigned_gt_inds[max_overlap_inds] = i + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_bboxes,), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, max_overlaps, labels=assigned_labels)

    def xy_wh_r_2_xy_sigma(self, xywhr):
        """Convert oriented bounding box to 2-D Gaussian distribution.

        Args:
            xywhr (torch.Tensor): rbboxes with shape (N, 5).

        Returns:
            xy (torch.Tensor): center point of 2-D Gaussian distribution
                with shape (N, 2).
            sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                with shape (N, 2, 2).
        """
        _shape = xywhr.shape
        assert _shape[-1] == 5
        xy = xywhr[..., :2]
        wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
        r = xywhr[..., 4]
        cos_r = torch.cos(r)
        sin_r = torch.sin(r)
        R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
        S = 0.5 * torch.diag_embed(wh)

        sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                                1)).reshape(_shape[:-1] + (2, 2))

        return xy, sigma


    def xy_wh_2_xy_sigma(self, xywh):
        """Convert horizontal bounding box to 2-D Gaussian distribution.

        Args:
            xywh (torch.Tensor): bboxes with shape (N, 4).

        Returns:
            xy (torch.Tensor): center point of 2-D Gaussian distribution
                with shape (N, 2).
            sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                with shape (N, 2, 2).
        """
        _shape = xywh.shape
        assert _shape[-1] == 4
        xy = xywh[..., :2]
        wh = xywh[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
        r = torch.zeros([_shape[0]]).type_as(xywh)
        cos_r = torch.cos(r)
        sin_r = torch.sin(r)
        R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
        S = 0.5 * torch.diag_embed(wh)

        sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                                1)).reshape(_shape[:-1] + (2, 2))

        return xy, sigma