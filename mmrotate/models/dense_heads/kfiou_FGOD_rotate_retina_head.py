# Copyright (c) SJTU. All rights reserved.
from ..builder import ROTATED_HEADS
from .rotated_retina_head import RotatedRetinaHead

import numpy as np
from scipy.linalg import orth
import torch.nn.functional as F
import torch.fft
import torch
import torch.nn as nn

from mmcv.runner import force_fp32
from mmdet.core import images_to_levels, multi_apply, unmap
from .rotated_base_dense_head import RotatedBaseDenseHead
from mmrotate.core import (aug_multiclass_nms_rotated, bbox_mapping_back,
                           build_assigner, build_bbox_coder,
                           build_prior_generator, build_sampler,
                           multiclass_nms_rotated, obb2hbb, obb2xyxy,
                           rotated_anchor_inside_flags)

import json
import os
from collections import defaultdict

# 全局变量用于存储统计信息
iter_stats = []
current_iter = 0


@ROTATED_HEADS.register_module()
class FGODKFIoURRetinaHead(RotatedRetinaHead):
    """Rotated Anchor-based head for KFIoU. The difference from `RRetinaHead`
    is that its loss_bbox requires bbox_pred, bbox_targets, pred_decode and
    targets_decode as inputs.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int, optional): Number of stacked convolutions.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        anchor_generator (dict): Config dict for anchor generator
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 with_frequency_aug=False,
                 with_orth_aug=False, 
                 high_freq_ratio=0.8,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 anchor_generator=dict(
                     type='AnchorGenerator',
                     octave_base_scale=4,
                     scales_per_octave=3,
                     ratios=[0.5, 1.0, 2.0],
                     strides=[8, 16, 32, 64, 128]),
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='retina_cls',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs):
        self.bboxes_as_anchors = None
        super(FGODKFIoURRetinaHead, self).__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            stacked_convs=stacked_convs,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            anchor_generator=anchor_generator,
            init_cfg=init_cfg,
            **kwargs)
        
        # module ablation
        self.with_frequency_aug = with_frequency_aug
        self.with_orth_aug = with_orth_aug
        if self.train_cfg:
            self.with_fgod_assign = 'FGOD' in self.train_cfg.assigner.type
        
        ### OM++
        self.num_om = self.num_anchors * self.cls_out_channels
        dim = self.feat_channels
        
        A = np.random.rand(dim, dim)
        orthogonal_basis = orth(A)
        new_value = torch.Tensor(orthogonal_basis)
        new_value = new_value / new_value.norm(dim=-1, keepdim=True)
        
        self.register_buffer('om', new_value[:self.num_om,:])
        
        self.new_value = nn.Parameter(new_value, requires_grad=True)
        
        ### fusion
        self.router = nn.Sequential(
            nn.Conv2d(dim, dim // 16, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 16, self.num_om, 1),
        )
        self.high_freq_ratio = high_freq_ratio

        self.iters = 0
    
    def forward(self, feats):
        return multi_apply(self.forward_single, feats[self.start_level:])
    
    def forward_single(self, x):
        N, C, H, W = x.shape
        device = x.device
        if self.with_frequency_aug:
            # RFFT
            x_freq = torch.fft.rfft(x, dim=1) # [N, C//2 + 1, H, W]
            
            # high/low frequency masking
            freq_len = x_freq.shape[1]
            cutoff = int(freq_len * (1 - self.high_freq_ratio))
            # High-Pass Mask
            mask_high = torch.ones((N, freq_len, H, W), device=device)
            mask_high[:, :cutoff, :, :] = 0 
            # Low-Pass Mask
            mask_low = 1 - mask_high
            x_freq_high = x_freq * mask_high
            x_freq_low = x_freq * mask_low
            
            # IRFFT
            x_high = torch.fft.irfft(x_freq_high, n=C, dim=1) # [N, C, H, W] 
            x_low = torch.fft.irfft(x_freq_low, n=C, dim=1)   # [N, C, H, W]
            
            x_high_norm = (x_high / x_high.norm(dim=1, keepdim=True))
            om_basis = self.om.to(device) # [class, C]
            
            projection_scores = torch.einsum('nchw, mc->nmhw', x_high_norm, om_basis)
            router_logits = self.router(x_low)
            gate_weights = router_logits.sigmoid()
            combined_coeff = projection_scores * gate_weights
            
            x_high_orthogonal_reconstructed = torch.einsum('nmhw, mc->nchw', combined_coeff, om_basis)
            x_mag = x_high.norm(p=2, dim=-1, keepdim=True)
            x_high_new = x_high_orthogonal_reconstructed * x_mag
            
            x_new = x_high_new + x_low
            cls_feat = x_new
            reg_feat = x_new
        else:
            cls_feat = x
            reg_feat = x
        
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        
        if self.with_orth_aug:
            cls_feat = cls_feat / cls_feat.norm(dim=1, keepdim=True)
            om_feats = self.om.to(cls_feat.dtype).to(cls_feat.device) + self.new_value.to(cls_feat.dtype).to(cls_feat.device)
            om_feats = om_feats / om_feats.norm(dim=-1, keepdim=True)
            cls_score = 30.0 * torch.einsum('nchw, mc->nmhw', cls_feat, om_feats)
        else:
            cls_score = self.retina_cls(cls_feat)

        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred
    
    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss. Default: None

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        self.iters += 1

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            cls_scores, 
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i, _ in enumerate(anchor_list):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
    
    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            cls_score (torch.Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (torch.Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 5, H, W).
            anchors (torch.Tensor): Box reference for each scale level with
                shape (N, num_total_anchors, 5).
            labels (torch.Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (torch.Tensor): Label weights of each anchor with
                shape (N, num_total_anchors)
            bbox_targets (torch.Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 5).
            bbox_weights (torch.Tensor): BBox regression loss weights of each
                anchor with shape (N, num_total_anchors, 5).
            num_total_samples (int): If sampling, num total samples equal to
                the number of total anchors; Otherwise, it is the number of
                positive anchors.

        Returns:
            tuple (torch.Tensor):

                - loss_cls (torch.Tensor): cls. loss for each scale level.
                - loss_bbox (torch.Tensor): reg. loss for each scale level.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)

        anchors = anchors.reshape(-1, 5)
        bbox_pred_decode = self.bbox_coder.decode(anchors, bbox_pred)
        bbox_targets_decode = self.bbox_coder.decode(anchors, bbox_targets)

        loss_bbox = self.loss_bbox(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            pred_decode=bbox_pred_decode,
            targets_decode=bbox_targets_decode,
            avg_factor=num_total_samples)
        return loss_cls, loss_bbox

    def _get_targets_single(self,
                            flat_anchors,
                            concat_scores, 
                            valid_flags,
                            gt_bboxes,
                            gt_bboxes_ignore,
                            gt_labels,
                            img_meta,
                            label_channels=1,
                            unmap_outputs=True):
        """Compute regression and classification targets for anchors in a
        single image.

        Args:
            flat_anchors (torch.Tensor): Multi-level anchors of the image,
                which are concatenated into a single tensor of shape
                (num_anchors, 5)
            valid_flags (torch.Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            gt_bboxes (torch.Tensor): Ground truth bboxes of the image,
                shape (num_gts, 5).
            img_meta (dict): Meta info of the image.
            gt_bboxes_ignore (torch.Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 5).
            img_meta (dict): Meta info of the image.
            gt_labels (torch.Tensor): Ground truth labels of each box,
                shape (num_gts,).
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple (list[Tensor]):

                - labels_list (list[Tensor]): Labels of each level
                - label_weights_list (list[Tensor]): Label weights of each \
                  level
                - bbox_targets_list (list[Tensor]): BBox targets of each level
                - bbox_weights_list (list[Tensor]): BBox weights of each level
                - num_total_pos (int): Number of positive samples in all images
                - num_total_neg (int): Number of negative samples in all images
        """
        inside_flags = rotated_anchor_inside_flags(
            flat_anchors, valid_flags, img_meta['img_shape'][:2],
            self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]
        scores = concat_scores[inside_flags, :]
        # import pdb; pdb.set_trace()
        if self.assign_by_circumhbbox is not None:
            if anchors.size(-1) == 4:
                gt_hbboxes = obb2xyxy(gt_bboxes, self.assign_by_circumhbbox)
            else:
                gt_hbboxes = obb2hbb(gt_bboxes, self.assign_by_circumhbbox)
            assign_result = self.assigner.assign(
                anchors, gt_hbboxes, gt_bboxes_ignore,
                None if self.sampling else gt_labels)
        else:
            if self.with_fgod_assign:
                assign_result = self.assigner.assign(
                    anchors, None, gt_bboxes, gt_bboxes_ignore,
                    None if self.sampling else gt_labels, 
                    scores)
            else:
                assign_result = self.assigner.assign(
                    anchors, gt_bboxes, gt_bboxes_ignore,
                    None if self.sampling else gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = anchors.new_zeros((anchors.size(0), self.reg_dim))
        bbox_weights = anchors.new_zeros((anchors.size(0), self.reg_dim))
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes

            # self.visualize_assignment(img_meta,
            #                           gt_labels,
            #                           gt_bboxes,
            #                           sampling_result, 
            #                           cls_score=scores,
            #                           vis_dir='/data/zhr/work_dirs_file/Remote_sensing_FGOD/visualization/MAR20_assignment_903')


            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class since v2.5.0
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = unmap(
                labels, num_total_anchors, inside_flags,
                fill=self.num_classes)  # fill bg label
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds, sampling_result)
    
    def anchor_center(self, anchors):
        # Get anchor centers from anchors.
        return anchors[..., :2]

    def visualize_assignment(self,
                             img_meta,
                             gt_labels,
                             gt_bboxes,
                             sampling_result,
                             cls_score=None,
                             vis_dir='work_dirs/assignment_vis'):
        import os
        import cv2
        import numpy as np
        import matplotlib.pyplot as plt
        from mmrotate.core.visualization.image import draw_rbboxes
        from mmcv import imread
        
        # Create visualization directory if not exists
        os.makedirs(vis_dir, exist_ok=True)
        
        # Get image path and name
        img_info = img_meta
        img_path = img_info.get('filename', f'unknown_img')
        img_name = os.path.basename(img_path)
        
        # Load image
        img = imread(img_path)
        if img is None:
            # If image path is not available, create a black image
            img_height, img_width = img_info['img_shape'][:2]
            img = np.zeros((img_height, img_width, 3), dtype=np.uint8)
        
        # Resize image to 800x800
        img = cv2.resize(img, (800, 800), interpolation=cv2.INTER_LINEAR)
            
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Extract GT bboxes for current image
        curr_gt_bboxes = gt_bboxes

        pos_inds = sampling_result.pos_inds
        pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds
        pos_anchors = sampling_result.pos_bboxes
        pos_assigned_labels = gt_labels[pos_assigned_gt_inds]

        # 1. 取唯一值（自动排序）
        unique_labels = torch.unique(pos_assigned_labels)

        # 2. 转为 python list
        unique_list = unique_labels.tolist()

        # 3. 转为字符串
        result_str = str(unique_list)


        # Get sampling result for current image
        cls_score = cls_score.reshape(-1).contiguous()
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(img_rgb)
        # ax.set_title(f'Label Assignment Visualization - Image {img_name}')
        
        # Draw GT boxes in green
        if len(curr_gt_bboxes) > 0:
            draw_rbboxes(ax, curr_gt_bboxes.cpu().numpy(), color='green', thickness=1, alpha=0.8)
        
        if len(pos_anchors) > 0:
            # Draw positive sample points (centers)
            pos_centers = self.anchor_center(pos_anchors).cpu().numpy()  # Extract (x, y) centers
            ax.scatter(pos_centers[:, 0], pos_centers[:, 1], 
                        c='red', s=30, marker='o', edgecolors='white', linewidth=0.5)
            
            # Draw positive sample regression boxes
            # draw_rbboxes(ax, pos_anchors.cpu().numpy(), color='red', thickness=1, alpha=0.6)
            
            # Add classification scores if available
            # Reshape and get scores for positive samples
            pos_scores = cls_score[pos_inds]
            
            # for i, (pos, scores, labels) in enumerate(zip(pos_anchors, pos_scores, pos_assigned_labels)):
            #     cx, cy = pos[0], pos[1]
            #     # Show max score for this anchor
            #     max_score = scores
            #     ax.text(cx, cy, f'{max_score:.2f}', 
            #             fontsize=8, color='red', 
            #             weight='bold',
            #             verticalalignment='bottom', 
            #             horizontalalignment='left')
        
        # Save the visualization
        vis_path = os.path.join(vis_dir, f'{img_name.split(".")[0]}_{result_str}_{self.iters}.jpg')
        plt.savefig(vis_path, bbox_inches='tight', dpi=150)
        plt.close()
            

    def get_targets(self,
                    anchor_list,
                    score_list, 
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True,
                    return_sampling_results=False):
        """Compute regression and classification targets for anchors in
        multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, 5).
            valid_flag_list (list[list[Tensor]]): Multi level valid flags of
                each image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
                ignored.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: Usually returns a tuple containing learning targets.

                - labels_list (list[Tensor]): Labels of each level.
                - label_weights_list (list[Tensor]): Label weights of each \
                    level.
                - bbox_targets_list (list[Tensor]): BBox targets of each level.
                - bbox_weights_list (list[Tensor]): BBox weights of each level.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.

            additional_returns: This function enables user-defined returns from
                `self._get_targets_single`. These returns are currently refined
                to properties at each feature map (i.e. having HxW dimension).
                The results will be concatenated after the end
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs
        
        ### cls_scores split
        cls_scores_anchor = [[0] * len(score_list)] * len(anchor_list)
        for level_id in range(len(score_list)):
            cls_score_level = score_list[level_id]
            cls_score_level_anchor = torch.split(cls_score_level, 1, dim=0)
            for anchor_ids in range(len(anchor_list)):
                cls_scores_anchor[anchor_ids][level_id] = torch.max(cls_score_level_anchor[anchor_ids].reshape(-1, score_list[0].shape[1]).sigmoid(), 
                                                                    dim=-1, keepdim=True)[0]
        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors to a single tensor
        concat_score_list = []
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            concat_score_list.append(torch.cat(cls_scores_anchor[i]))
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        
        # # 用于收集统计信息
        # all_class_positive_counts = defaultdict(int)
        # all_target_positive_counts = []
        
        results = multi_apply(
            self._get_targets_single,
            concat_anchor_list,
            concat_score_list, 
            concat_valid_flag_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs)
        (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
         pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
        rest_results = list(results[7:])  # user-added return values
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        
        # # 计算统计信息
        # for i, sampling_result in enumerate(sampling_results_list):
        #     if sampling_result is not None:
        #         pos_inds = sampling_result.pos_inds
        #         if len(pos_inds) > 0:
        #             # 获取正样本对应的GT标签
        #             pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds
        #             if (gt_labels_list is not None and 
        #                 i < len(gt_labels_list)):
        #                 current_gt_labels = gt_labels_list[i]
        #                 if (current_gt_labels is not None and 
        #                     pos_assigned_gt_inds is not None and 
        #                     len(pos_assigned_gt_inds) > 0 and 
        #                     current_gt_labels.numel() > 0):
        #                     # 使用分配的GT索引来获取对应的标签
        #                     try:
        #                         # 确保索引不超过范围
        #                         if current_gt_labels.size(0) > 0:
        #                             valid_indices_mask = pos_assigned_gt_inds < current_gt_labels.size(0)
        #                             valid_indices = pos_assigned_gt_inds[valid_indices_mask]
        #                             if valid_indices.numel() > 0:
        #                                 pos_gt_labels = current_gt_labels[valid_indices.long()]
                                        
        #                                 # 统计每个类别的正样本数量
        #                                 unique_labels, counts = torch.unique(pos_gt_labels, return_counts=True)
        #                                 for label, count in zip(unique_labels.cpu().numpy(), counts.cpu().numpy()):
        #                                     all_class_positive_counts[int(label)] += int(count)
        #                     except:
        #                         # 如果出现索引错误，则跳过
        #                         pass
                    
        #             # 统计每个目标分配的正样本数量
        #             if (gt_labels_list is not None and 
        #                 i < len(gt_labels_list) and 
        #                 pos_assigned_gt_inds is not None):
        #                 current_gt_labels = gt_labels_list[i]
        #                 if current_gt_labels is not None:
        #                     try:
        #                         unique_gt_inds, gt_counts = torch.unique(pos_assigned_gt_inds, return_counts=True)
        #                         for gt_idx, count in zip(unique_gt_inds.cpu().numpy(), gt_counts.cpu().numpy()):
        #                             if current_gt_labels.size(0) > gt_idx:  # 确保索引有效
        #                                 gt_label = int(current_gt_labels[gt_idx].item())
        #                                 all_target_positive_counts.append({
        #                                     'image_index': i,
        #                                     'target_index': int(gt_idx),
        #                                     'class': gt_label,
        #                                     'positive_count': int(count)
        #                                 })
        #                     except:
        #                         # 如果出现索引错误，则跳过
        #                         pass

        # # 保存统计信息到全局变量
        # global iter_stats, current_iter
        # stats_entry = {
        #     'iter': current_iter,
        #     'class_positive_counts': dict(all_class_positive_counts),
        #     'target_positive_counts': all_target_positive_counts
        # }
        # iter_stats.append(stats_entry)
        
        # # 每隔一定iter保存一次到文件
        # if current_iter % 10 == 0:  # 每100个iter保存一次
        #     # os.makedirs('work_dirs', exist_ok=True)
        #     with open(f'/data/zhr/work_dirs_file/Remote_sensing_FGOD/retinanet_kfiou_ours_mar20_3x/training_stats.json', 'w') as f:
        #         json.dump(iter_stats, f, indent=2)
        
        # current_iter += 1
        
        
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        res = (labels_list, label_weights_list, bbox_targets_list,
               bbox_weights_list, num_total_pos, num_total_neg)
        if return_sampling_results:
            res = res + (sampling_results_list, )
        for i, r in enumerate(rest_results):  # user-added return values
            rest_results[i] = images_to_levels(r, num_level_anchors)

        return res + tuple(rest_results)

    def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): Device for returned tensors

        Returns:
            tuple (list[Tensor]):

                - anchor_list (list[Tensor]): Anchors of each image.
                - valid_flag_list (list[Tensor]): Valid flags of each image.
        """
        num_imgs = len(img_metas)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.anchor_generator.grid_priors(
            featmap_sizes, device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = self.anchor_generator.valid_flags(
                featmap_sizes, img_meta['pad_shape'], device)
            valid_flag_list.append(multi_level_flags)

        return anchor_list, valid_flag_list