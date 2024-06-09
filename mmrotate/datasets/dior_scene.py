import os.path as osp
import xml.etree.ElementTree as ET

import mmcv
from mmcv import print_log

from mmdet.datasets.xml_style_dior import XMLDIORDataset as XMLDataset
import numpy as np
from collections import OrderedDict

from mmrotate.core.evaluation import eval_rbbox_recalls
from mmrotate.core import eval_rbbox_map, poly2obb_np
from .builder import ROTATED_DATASETS


@ROTATED_DATASETS.register_module()
class DIOR_SCENEDataset(XMLDataset):
    CLASSES = ('airport', 'harbor')

    def __init__(self,
                 version='oc',
                 **kwargs):
        self.version = version
        super(DIOR_SCENEDataset, self).__init__(**kwargs)

    def get_ann_info(self, idx):
        """Get annotation from XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        xml_path = osp.join(self.ann_subdir, f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        polygons = []
        bboxes_ignore = []
        labels_ignore = []
        polygons_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.CLASSES:
                continue
            label = self.cat2label[name]
            difficult = obj.find('difficult')
            difficult = 0 if difficult is None else int(difficult.text)
            bnd_box = obj.find('robndbox')
            # TODO: check whether it is necessary to use int
            # Coordinates may be float type
            polygon = [
                float(bnd_box.find('x_left_top').text),
                float(bnd_box.find('y_left_top').text),
                float(bnd_box.find('x_right_top').text),
                float(bnd_box.find('y_right_top').text),
                float(bnd_box.find('x_right_bottom').text),
                float(bnd_box.find('y_right_bottom').text),
                float(bnd_box.find('x_left_bottom').text),
                float(bnd_box.find('y_left_bottom').text)
            ]
            polygon = np.array(polygon, dtype=np.float32)
            bbox = np.array(poly2obb_np(
                polygon, self.version), dtype=np.float32)
            if bbox.size != 5:
                continue
            ignore = False
            if self.min_size:
                assert not self.test_mode
                w, h = bbox[2], bbox[3]
                if w < self.min_size or h < self.min_size:
                    ignore = True
            if difficult or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
                polygons_ignore.append(polygon)
            else:
                bboxes.append(bbox)
                labels.append(label)
                polygons.append(polygon)
        if not bboxes:
            bboxes = np.zeros((0, 5))
            labels = np.zeros((0, ))
            polygons = np.zeros((0, 8))
        else:
            bboxes = np.array(bboxes, ndmin=2)
            labels = np.array(labels)
            polygons = np.array(polygons)
            # bboxes = np.array(bboxes)
            # if (len(bboxes.shape) == 2 and bboxes.shape[1] == 11) or (len(bboxes.shape) == 1 and bboxes.shape[0] == 11):
            #     bboxes = np.zeros((0, 5))
            #     labels = np.zeros((0, ))
            #     polygons = np.zeros((0, 8))
            # else:
            #     bboxes = bboxes.reshape((-1,5))
            #     labels = np.array(labels)
            #     polygons = np.array(polygons)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 5))
            labels_ignore = np.zeros((0, ))
            polygons_ignore = np.zeros((0, 8))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2)
            labels_ignore = np.array(labels_ignore)
            polygons_ignore = np.array(polygons_ignore)
        
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            polygons=polygons.astype(np.float32),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64),
            polygons_ignore=polygons_ignore.astype(np.float32))
        return ann

    def evaluate(
            self,
            results,
            metric='mAP',
            logger=None,
            proposal_nums=(100, 300, 500, 1000, 2000),
            iou_thr=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
            scale_ranges=None,
            use_07_metric=True,
            nproc=4):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
            use_07_metric (bool): Whether to use the voc07 metric.
            nproc (int): Processes used for computing TP and FP.
                Default: 4.
        """
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP', 'recall']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')

        annotations = [self.get_ann_info(i) for i in range(len(self))]
        eval_results = OrderedDict()
        iou_thrs = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        if metric == 'mAP':
            assert isinstance(iou_thrs, list)
            mean_aps = []
            for iou_thr in iou_thrs:
                print_log(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                mean_ap, _ = eval_rbbox_map(
                    results,
                    annotations,
                    scale_ranges=scale_ranges,
                    iou_thr=iou_thr,
                    use_07_metric=use_07_metric,
                    dataset=self.CLASSES,
                    logger=logger,
                    nproc=nproc)
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 4)
            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            eval_results.move_to_end('mAP', last=False)
        elif metric == 'recall':
            assert mmcv.is_list_of(results, np.ndarray)
            gt_bboxes = []
            for i in range(len(self)):
                ann = self.get_ann_info(i)
                bboxes = ann['bboxes']
                gt_bboxes.append(bboxes)
            if isinstance(iou_thr, float):
                iou_thr = [iou_thr]
            recalls = eval_rbbox_recalls(
                gt_bboxes, results, proposal_nums, iou_thr, logger=logger)
            for i, num in enumerate(proposal_nums):
                for j, iou in enumerate(iou_thr):
                    eval_results[f'recall@{num}@{iou}'] = recalls[i, j]
            if recalls.shape[1] > 1:
                ar = recalls.mean(axis=1)
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
        else:
            raise NotImplementedError

        return eval_results
