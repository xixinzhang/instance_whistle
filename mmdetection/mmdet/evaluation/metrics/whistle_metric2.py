# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import itertools
import os.path as osp
import tempfile
from collections import OrderedDict, defaultdict, deque
from time import time
from typing import Dict, List, Optional, Sequence, Union
from rich import print as rprint
from skimage.morphology import skeletonize
from copy import deepcopy

import numpy as np
import torch
import copy
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump, get_local_path, load
from mmengine.logging import MMLogger
from terminaltables import AsciiTable

from mmdet.datasets.api_wrappers import COCO, COCOeval, COCOevalMP
from mmdet.registry import METRICS
from mmdet.structures.mask import encode_mask_results
from tqdm import tqdm
from ..functional import eval_recalls

import pycocotools.mask as maskUtils
from rich import print as rprint

@METRICS.register_module()
class WhistleMetric2(BaseMetric):
    """Whistle evaluation metric.

    Evaluate AR, AP, and mAP for detection tasks including proposal/box
    detection and instance segmentation. Please refer to
    https://cocodataset.org/#detection-eval for more details.

    Args:
        ann_file (str, optional): Path to the coco format annotation file.
            If not specified, ground truth annotations from the dataset will
            be converted to coco format. Defaults to None.
        metric (str | List[str]): Metrics to be evaluated. Valid metrics
            include 'bbox', 'segm', 'proposal', and 'proposal_fast'.
            Defaults to 'bbox'.
        classwise (bool): Whether to evaluate the metric class-wise.
            Defaults to False.
        proposal_nums (Sequence[int]): Numbers of proposals to be evaluated.
            Defaults to (100, 300, 1000).
        iou_thrs (float | List[float], optional): IoU threshold to compute AP
            and AR. If not specified, IoUs from 0.5 to 0.95 will be used.
            Defaults to None.
        metric_items (List[str], optional): Metric result names to be
            recorded in the evaluation result. Defaults to None.
        format_only (bool): Format the output results without perform
            evaluation. It is useful when you want to format the result
            to a specific format and submit it to the test server.
            Defaults to False.
        outfile_prefix (str, optional): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Defaults to None.
        file_client_args (dict, optional): Arguments to instantiate the
            corresponding backend in mmdet <= 3.0.0rc6. Defaults to None.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
        sort_categories (bool): Whether sort categories in annotations. Only
            used for `Objects365V1Dataset`. Defaults to False.
        use_mp_eval (bool): Whether to use mul-processing evaluation
    """
    default_prefix: Optional[str] = 'whistle2'

    def __init__(self,
                 ann_file: Optional[str] = None,
                 metric: Union[str, List[str]] = 'bbox',
                 classwise: bool = False,
                 proposal_nums: Sequence[int] = (100, 300, 1000),
                 iou_thrs: Optional[Union[float, Sequence[float]]] = None,
                 metric_items: Optional[Sequence[str]] = None,
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 file_client_args: dict = None,
                 backend_args: dict = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 sort_categories: bool = False,
                 use_mp_eval: bool = False) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        # coco evaluation metrics
        self.metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in self.metrics:
            if metric not in allowed_metrics:
                raise KeyError(
                    "metric should be one of 'bbox', 'segm', 'proposal', "
                    f"'proposal_fast', but got {metric}.")

        # do class wise evaluation, default False
        self.classwise = classwise
        # whether to use multi processing evaluation, default False
        self.use_mp_eval = use_mp_eval

        # proposal_nums used to compute recall or precision.
        self.proposal_nums = list(proposal_nums)

        # iou_thrs used to compute recall or precision.
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        self.iou_thrs = iou_thrs
        self.metric_items = metric_items
        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, 'outfile_prefix must be not'
            'None when format_only is True, otherwise the result files will'
            'be saved to a temp directory which will be cleaned up at the end.'

        self.outfile_prefix = outfile_prefix

        self.backend_args = backend_args
        if file_client_args is not None:
            raise RuntimeError(
                'The `file_client_args` is deprecated, '
                'please use `backend_args` instead, please refer to'
                'https://github.com/open-mmlab/mmdetection/blob/main/configs/_base_/datasets/coco_detection.py'  # noqa: E501
            )

        # if ann_file is not specified,
        # initialize coco api with the converted dataset
        if ann_file is not None:
            with get_local_path(
                    ann_file, backend_args=self.backend_args) as local_path:
                self._coco_api = COCO(local_path)
                if sort_categories:
                    # 'categories' list in objects365_train.json and
                    # objects365_val.json is inconsistent, need sort
                    # list(or dict) before get cat_ids.
                    cats = self._coco_api.cats
                    sorted_cats = {i: cats[i] for i in sorted(cats)}
                    self._coco_api.cats = sorted_cats
                    categories = self._coco_api.dataset['categories']
                    sorted_categories = sorted(
                        categories, key=lambda i: i['id'])
                    self._coco_api.dataset['categories'] = sorted_categories
        else:
            self._coco_api = None

        # handle dataset lazy init
        self.cat_ids = None
        self.img_ids = None


    def results2whistles(self, results: Sequence[dict],
                    filter_dt = 0.9) -> dict:
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (Sequence[dict]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict: Possible keys are "bbox", "segm", "proposal", and
            values are corresponding filenames.
        """
        if  'masks' not in results[0]:
            raise ValueError('The results should contain masks')
        
        whistles_json = [] if 'masks' in results[0] else None
        for idx, result in enumerate(tqdm(results, desc='Convert dt to whistle')):
            image_id = result.get('img_id', idx)
            labels = result['labels']
            scores = result['scores']

            # segm results
            masks = result['masks']
            mask_scores = result.get('mask_scores', scores)
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['score'] = float(mask_scores[i])
                if data['score'] <  filter_dt:
                    continue
                data['category_id'] = self.cat_ids[label]
                if isinstance(masks[i]['counts'], bytes):
                    masks[i]['counts'] = masks[i]['counts'].decode()
                bitmap = maskUtils.decode(masks[i])
                if bitmap.sum() > 0:
                    data['whistle'] = mask_to_whistle(bitmap)
                else:
                    continue
                whistles_json.append(data)

        rprint(f'Num of detected whistles: {len(whistles_json)} filtered by score {filter_dt}')

        # TODO: dump whistles

        # result_files['bbox'] = f'{outfile_prefix}.bbox.json'
        # result_files['proposal'] = f'{outfile_prefix}.bbox.json'
        # dump(bbox_json_results, result_files['bbox'])

        # if segm_json_results is not None:
        #     result_files['segm'] = f'{outfile_prefix}.segm.json'
        #     dump(segm_json_results, result_files['segm'])

        return whistles_json

    # TODO: data_batch is no longer needed, consider adjusting the
    #  parameter position
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of data samples that
                contain annotations and predictions.
        """
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()
            # encode mask to RLE
            if 'masks' in pred:
                result['masks'] = encode_mask_results(
                    pred['masks'].detach().cpu().numpy()) if isinstance(
                        pred['masks'], torch.Tensor) else pred['masks']
            # some detectors use different scores for bbox and mask
            if 'mask_scores' in pred:
                result['mask_scores'] = pred['mask_scores'].cpu().numpy()

            # parse gt
            gt = dict()
            gt['width'] = data_sample['ori_shape'][1]
            gt['height'] = data_sample['ori_shape'][0]
            gt['img_id'] = data_sample['img_id']
            if self._coco_api is None:
                # TODO: Need to refactor to support LoadAnnotations
                assert 'instances' in data_sample, \
                    'ground truth is required for evaluation when ' \
                    '`ann_file` is not provided'
                gt['anns'] = data_sample['instances']
            # add converted result to the results list
            self.results.append((gt, result))

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # split gt and prediction list
        _, preds = zip(*results)

        # add img_id in same audio file
        if self._coco_api is not None:
            audio_to_img = defaultdict(list)
            audio_to_frame = defaultdict(list)
            for id, img in self._coco_api.imgs.items():
                audio_to_img[img['audio_filename']].append(id)
                audio_to_frame[img['start_frame']].append(id)
            
            gt_per_img = []
            for id, ann in tqdm(self._coco_api.anns.items(), desc='Convert gt to whistle'):
                gt_per_img.append({
                    'image_id': ann['image_id'],
                    'score': 1.0,
                    'category_id': ann['category_id'],
                    'whistle': mask_to_whistle(
                        self._coco_api.annToMask(ann))
            })
            rprint(f'Num of gt whistles: {len(gt_per_img)} from {len(results)} imgs')
            

        # handle lazy init
        if self.cat_ids is None:
            self.cat_ids = self._coco_api.get_cat_ids(
                cat_names=self.dataset_meta['classes'])
        if self.img_ids is None:
            self.img_ids = self._coco_api.get_img_ids()

        # convert predictions to coco format and dump to json file
        dt_per_img = self.results2whistles(preds, filter_dt=0.1)

        w = results[0][0]['width']
        img_to_dts = defaultdict(list)
        for i, dt in enumerate(dt_per_img):
            img_to_dts[dt['image_id']].append(
                dt['whistle'],
            )
        
        img_to_gts = defaultdict(list)
        for i, gt in enumerate(gt_per_img):
            img_to_gts[gt['image_id']].append(
                gt['whistle'],
            )

        img_to_whistles = dict()
        for img_id, _ in self._coco_api.imgs.items():
            img_to_whistles[img_id] = {
                'gts': img_to_gts[img_id],
                'dts': img_to_dts[img_id],
                'w': w,
                'img_id': img_id,
            }


        eval_results = OrderedDict()
        

        metric = self.metrics[0]
        logger.info(f'Evaluating {metric}...')
        res = accumulate_wistle_results(img_to_whistles, valid_gt=True)
        summary = sumerize_whisle_results(res)
        rprint(summary)

        return eval_results



def get_traj_valid(conf_map, traj):

    conf_traj = conf_map[traj[:, 1], traj[:, 0]]
    score = conf_traj.mean()
    sorted_idx = np.argsort(conf_traj)
    ratio = 0.3
    if len(sorted_idx) > 0:
        bound = conf_traj[sorted_idx[int(len(sorted_idx) * ratio)]]
        return score, bound
    else:
        raise ValueError("No valid points in the trajectory")


def bresenham_line(p1, p2):
    """
    Implements Bresenham's line algorithm for grid-aligned paths.
    This creates the most direct path between two points while staying on the grid.
    
    Args:
        p1: Starting point (x1, y1)
        p2: Ending point (x2, y2)
        
    Returns:
        List of points [(x1,y1), ..., (x2,y2)] forming a continuous path
    """
    x1, y1 = p1
    x2, y2 = p2
    
    # Initialize the path with the starting point
    path = []
    
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    while True:
        path.append((x1, y1))
        
        if x1 == x2 and y1 == y2:
            break
            
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    
    return path


def midpoint_interpolation(p1, p2):
    """
    Recursively creates a path by adding midpoints between points.
    
    Args:
        p1: Starting point (x1, y1)
        p2: Ending point (x2, y2)
        
    Returns:
        List of points forming a continuous path
    """
    x1, y1 = p1
    x2, y2 = p2
    
    # Base case: points are adjacent or the same
    if abs(x2 - x1) <= 1 and abs(y2 - y1) <= 1:
        return [p1, p2]
    
    # Find the midpoint (rounded to integers)
    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2
    midpoint = (mid_x, mid_y)
    
    # Recursively find paths for each half
    first_half = midpoint_interpolation(p1, midpoint)
    second_half = midpoint_interpolation(midpoint, p2)
    
    # Combine the paths (avoid duplicating the midpoint)
    return first_half[:-1] + second_half


def simple_weighted_path(points, i, window=2):
    """
    A simpler weighted path method that doesn't use BFS.
    Uses direct interpolation with local direction awareness.
    
    Args:
        points: All trajectory points
        i: Current index
        window: Number of points to consider on each side
        
    Returns:
        List of points forming a continuous path
    """
    # Ensure all points are tuples
    points = [tuple(p) for p in points]
    current = points[i]
    next_point = points[i + 1]
    
    # If points are close, use Bresenham's algorithm
    if abs(next_point[0] - current[0]) <= 3 and abs(next_point[1] - current[1]) <= 3:
        return bresenham_line(current, next_point)
    
    # Get neighboring points within window
    start_idx = max(0, i - window)
    end_idx = min(len(points) - 1, i + 1 + window)
    neighbors = points[start_idx:end_idx + 1]
    
    # Simple weighted path based on direction awareness
    if len(neighbors) <= 2:
        return bresenham_line(current, next_point)
    
    # Calculate primary direction from neighbors
    x_coords = [p[0] for p in neighbors]
    y_coords = [p[1] for p in neighbors]
    
    # Simple linear trend (direction)
    if len(neighbors) >= 3:
        x_diffs = [x_coords[j+1] - x_coords[j] for j in range(len(x_coords)-1)]
        y_diffs = [y_coords[j+1] - y_coords[j] for j in range(len(y_coords)-1)]
        avg_x_diff = sum(x_diffs) / len(x_diffs)
        avg_y_diff = sum(y_diffs) / len(y_diffs)
    else:
        avg_x_diff = next_point[0] - current[0]
        avg_y_diff = next_point[1] - current[1]
    
    # Create a path with awareness of the typical step size
    path = [current]
    
    # Current position
    cx, cy = current
    tx, ty = next_point
    
    # Step sizes (make them integers between 1-3 based on average direction)
    step_x = max(1, min(3, int(abs(avg_x_diff)) or 1)) * (1 if tx > cx else -1 if tx < cx else 0)
    step_y = max(1, min(3, int(abs(avg_y_diff)) or 1)) * (1 if ty > cy else -1 if ty < cy else 0)
    
    # We'll adjust step_x and step_y to ensure we don't overshoot
    while (cx, cy) != next_point:
        # Decide which direction to move
        if abs(cx - tx) > abs(cy - ty):
            # Move in x direction
            new_cx = cx + step_x
            # Check if we'd overshoot
            if (step_x > 0 and new_cx > tx) or (step_x < 0 and new_cx < tx):
                new_cx = tx
            cx = new_cx
        else:
            # Move in y direction
            new_cy = cy + step_y
            # Check if we'd overshoot
            if (step_y > 0 and new_cy > ty) or (step_y < 0 and new_cy < ty):
                new_cy = ty
            cy = new_cy
        
        # Add new point to path
        new_point = (cx, cy)
        
        # Check if we need to fill gaps (ensure path is grid-continuous)
        last_point = path[-1]
        if abs(new_point[0] - last_point[0]) > 1 or abs(new_point[1] - last_point[1]) > 1:
            # Fill gap with Bresenham
            gap_filler = bresenham_line(last_point, new_point)
            path.extend(gap_filler[1:])
        else:
            path.append(new_point)
    
    return path


def bezier_grid_path(points, start_idx, end_idx, steps=10):
    """
    Creates a Bezier curve between points and snaps it to grid.
    
    Args:
        points: All trajectory points
        start_idx: Index of starting point
        end_idx: Index of ending point
        steps: Number of interpolation steps
        
    Returns:
        List of grid points approximating a Bezier curve
    """
    # Extract points
    p0 = points[start_idx]
    p3 = points[end_idx]
    
    # Use neighboring points to determine control points if available
    if start_idx > 0 and end_idx < len(points) - 1:
        # Control points based on neighboring points
        prev = points[start_idx - 1]
        next_point = points[end_idx + 1]
        
        # Create control points by extending the lines from neighbors
        dx1 = p0[0] - prev[0]
        dy1 = p0[1] - prev[1]
        p1 = (p0[0] + dx1 // 2, p0[1] + dy1 // 2)
        
        dx2 = p3[0] - next_point[0]
        dy2 = p3[1] - next_point[1]
        p2 = (p3[0] + dx2 // 2, p3[1] + dy2 // 2)
    else:
        # Default control points for endpoints
        dx = p3[0] - p0[0]
        dy = p3[1] - p0[1]
        p1 = (p0[0] + dx // 3, p0[1] + dy // 3)
        p2 = (p0[0] + 2 * dx // 3, p0[1] + 2 * dy // 3)
    
    # Generate Bezier curve points
    curve_points = []
    for t in np.linspace(0, 1, steps):
        # Cubic Bezier formula
        x = (1-t)**3 * p0[0] + 3*(1-t)**2*t * p1[0] + 3*(1-t)*t**2 * p2[0] + t**3 * p3[0]
        y = (1-t)**3 * p0[1] + 3*(1-t)**2*t * p1[1] + 3*(1-t)*t**2 * p2[1] + t**3 * p3[1]
        
        # Round to nearest grid point
        curve_points.append((round(x), round(y)))
    
    # Ensure the path is continuous by filling any gaps
    grid_path = [p0]
    for i in range(1, len(curve_points)):
        prev = grid_path[-1]
        current = curve_points[i]
        
        # If points aren't adjacent, fill the gap with Bresenham
        if abs(current[0] - prev[0]) > 1 or abs(current[1] - prev[1]) > 1:
            connecting_points = bresenham_line(prev, current)
            grid_path.extend(connecting_points[1:])
        else:
            grid_path.append(current)
    
    # Ensure end point is included
    if grid_path[-1] != p3:
        connecting_points = bresenham_line(grid_path[-1], p3)
        grid_path.extend(connecting_points[1:])
    
    return grid_path

def mask_to_whistle(mask, method='bresenham'):
    """convert the instance mask to whistle contour, use skeleton methods
    
    Args
        mask: instance mask (H, W)
    Return
        whistle: (N,2) in pixel coordinates
    """
    mask = mask.astype(np.uint8)
    skeleton = skeletonize(mask).astype(np.uint8)
    whistle = np.array(np.nonzero(skeleton)).T # [(y, x]
    whistle = np.flip(whistle, axis=1)  # [(x, y)]
    whistle = np.unique(whistle, axis=0)  # remove duplicate points
    whistle = whistle[whistle[:, 0].argsort()]
    assert whistle.ndim ==2 and whistle.shape[1] == 2, f"whistle shape: {whistle.shape}"

    # connect fragmented whistle points
    whistle = whistle.tolist()
    whistle_ =[whistle[0]]
    for i in range(len(whistle) - 1):
        current = whistle[i]
        next_point = whistle[i + 1]
        
        # Generate intermediate points between current and next_point
        if method == 'bresenham':
            intermediate_points = bresenham_line(current, next_point)
        elif method == 'midpoint':
            intermediate_points = midpoint_interpolation(current, next_point)
        elif method == 'bezier':
            intermediate_points = bezier_grid_path(whistle, i, i+1)
        elif method == 'weighted':
            intermediate_points = simple_weighted_path(whistle, i)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Add intermediate points to trajectory (skip the first one as it's already included)
        whistle_.extend(intermediate_points[1:])
    
    # Group by x-coordinate and select one y-value per x
    whistle_ = np.array(whistle_)
    unique_x = np.unique(whistle_[:, 0])
    averaged_y = np.zeros_like(unique_x)
    for i, x in enumerate(unique_x):
        y_values = whistle_[whistle_[:, 0] == x][:, 1]
        averaged_y[i] = int(np.round(np.mean(y_values)))
    unique_whistle = np.column_stack((unique_x, averaged_y))

    return unique_whistle

def gather_whistles(coco_gt:COCO, coco_dt:COCO, filter_dt=0.95, valid_gt=False, root_dir=None, debug=False):
    """gather per image whistles from instance masks"""
    
    if debug:
        coco_dt = deepcopy(coco_gt)
        for ann in coco_dt.anns.values():
            ann['score'] = 1.0

    img_to_whistles = dict()
    for img_id in coco_dt.imgs.keys():
        img = coco_dt.imgs[img_id]
        h, w = img['height'], img['width']
        dt_anns = coco_dt.imgToAnns[img_id]
        if debug and len(dt_anns) == 0:  # keep gt
            continue
        gt_anns = coco_gt.imgToAnns[img_id]
        gt_masks = [coco_gt.annToMask(ann) for ann in gt_anns]

        dt_masks = []
        for i, ann in enumerate(dt_anns):
            score = ann['score']
            mask = coco_dt.annToMask(ann)
            if score > filter_dt:
                dt_masks.append(mask)
            else:
                continue

        gt_whistles = [mask_to_whistle(mask) for mask in gt_masks]
        
        # bounds = []
        # gt_whistles_ = []
        # img_info = coco_gt.imgs[img_id]
        # image = cv2.imread(os.path.join(root_dir,'coco/val/data' , img_info['file_name']))
        # for whistle in gt_whistles:
        #     whistle = np.array(whistle)
        #     score, bound = get_traj_valid(image[...,0], whistle)
        #     bounds.append(bound)
        #     gt_whistles_.append(whistle)
        # gt_whistles = gt_whistles_

        dt_whistles = [mask_to_whistle(mask) for mask in dt_masks if mask.sum()>0]
        
        if debug:
            assert len(gt_whistles) == len(dt_whistles), f"gt and dt should have the \
            same number of whistles, gt: {len(gt_whistles)}, dt: {len(dt_whistles)}"
        
        img_to_whistles[img['id']] = {
            'gts': gt_whistles,
            # 'boudns_gt': bounds,
            # 'boudns_gt': None,
            'dts': dt_whistles,
            'w': w,
            'img_id': img_id,
        }
    sum_gts = sum([len(whistles['gts']) for whistles in img_to_whistles.values()])
    sum_dts = sum([len(whistles['dts']) for whistles in img_to_whistles.values()])
    rprint(f'gathered {len(img_to_whistles)} images with {sum_gts} gt whistles, {sum_dts} dt whistles')
    return img_to_whistles


def compare_whistles(gts, dts, w, img_id, boudns_gt=None, valid_gt = False, debug=False):
    """given whistle gt and dt in evaluation unit and get comparison results
    Args:
        gts, dts: N, 2 in format of y, x(or t, f)
    """
    gt_num = len(gts)
    # boudns_gt = np.array(boudns_gt)
    gt_ranges = np.zeros((gt_num, 2))
    gt_durations = np.zeros(gt_num)

    dt_num = len(dts)
    dt_ranges = np.zeros((dt_num, 2))
    dt_durations = np.zeros(dt_num)
    
    if debug:
        # for i in range(dt_num):
        #    assert (gts[i]== dts[i]).all(), f"gt and dt should not be the same {len(gts)} vs {len(dts)}"
        pass

    for gt_idx, gt in enumerate(gts):
        gt_start_x = max(0, gt[:, 0].min())
        gt_end_x = min(w -1 , gt[:, 0].max())
        gt_dura = gt_end_x + 1 - gt_start_x  # add 1 in pixel
        gt_durations[gt_idx] = gt_dura
        gt_ranges[gt_idx] = (gt_start_x, gt_end_x)
  
    for dt_idx, dt in enumerate(dts):
        dt_start_x = max(0, dt[:, 0].min())
        dt_end_x = min(w, dt[:, 0].max())
        dt_ranges[dt_idx] = (dt_start_x, dt_end_x)
        dt_durations[dt_idx] = dt_end_x + 1 - dt_start_x # add 1 in pixel
    
    dt_false_pos_all = list(range(dt_num))
    dt_true_pos_all = []
    dt_true_pos_valid = []
    gt_matched_all = []
    gt_matched_valid = []
    gt_missed_all = []
    gt_missed_valid = []
    all_deviation = []
    all_covered = []
    all_dura = []

    # go through each ground truth
    for gt_idx, gt in enumerate(gts):
        gt_start_x, gt_end_x = gt_ranges[gt_idx]
        gt_dura = gt_durations[gt_idx]

        # if gt_dura < 2:
        #     continue

        # Note: remove interpolation and snr validation that filter low gt snr level
        # which requires addition input
        dt_start_xs, dt_end_xs = dt_ranges[:, 0], dt_ranges[:, 1]
        ovlp_cond = (dt_start_xs <= gt_start_x) & (dt_end_xs >= gt_start_x) \
                        | (dt_start_xs>= gt_start_x) & (dt_start_xs <= gt_end_x)
        ovlp_dt_ids = np.nonzero(ovlp_cond)[0]

        matched = False
        deviations = []
        covered = 0
        # cnt = 0
        # dt_matched = []
        # dt_matched_dev = []
        # Note remove duration filter short < 75 pix whistle

        valid = True
        if valid_gt:
            if gt_dura < 75: # or boudns_gt[gt_idx] < 3:
                valid= False
        # rprint(f'valid:{valid}, gt_dura: {gt_dura}, boudns_gt: {boudns_gt[gt_idx]}')


        for ovlp_dt_idx in ovlp_dt_ids:
            ovlp_dt = dts[ovlp_dt_idx]
            dt_xs, dt_ys = ovlp_dt[:, 0], ovlp_dt[:, 1]
            dt_ovlp_x_idx = np.nonzero((dt_xs >= gt_start_x) & (dt_xs <= gt_end_x))[0]
            dt_ovlp_xs = dt_xs[dt_ovlp_x_idx]
            dt_ovlp_ys = dt_ys[dt_ovlp_x_idx]

            # Note: remove interpolation
            gt_ovlp_ys = gt[:, 1][np.searchsorted(gt[:, 0], dt_ovlp_xs)]
            deviation = np.abs(gt_ovlp_ys - dt_ovlp_ys)
            deviation_tolerence = 350 / 125
            if debug:
                # deviation_tolerence = 0.1
                pass
            if len(deviation)> 0 and np.mean(deviation) <= deviation_tolerence:
                matched = True
                
                # cnt += 1
                # dt_matched.append(ovlp_dt_idx)
                if ovlp_dt_idx in dt_false_pos_all:
                    dt_false_pos_all.remove(ovlp_dt_idx)
                # TODO: has multiplications
                dt_true_pos_all.append(ovlp_dt_idx)  
                # TODO: the deviation and coverage of same overlap can be counted multiple times
                deviations.extend(deviation)

                # dt_matched_dev.append(deviation)
                
                covered += dt_ovlp_xs.max() - dt_ovlp_xs.min() + 1
                if valid:
                    dt_true_pos_valid.append(ovlp_dt_idx)


        # if debug and cnt > 1:
        #     rprint(f"img_id: {img_id}, multiplication gt_id: {gt_idx} cnt: {cnt}")
        #     rprint(f"gt: {gt.T}",)
        #     for i, idx in enumerate(dt_matched):
        #         rprint(f"dt_idx: {idx}")
        #         rprint(f"dt: {dts[idx].T}")
        #         print(f"deviation: {dt_matched_dev[i].mean()}")
        
        if matched:
            gt_matched_all.append(gt_idx)
            if valid:
                gt_matched_valid.append(gt_idx)
        else:
            gt_missed_all.append(gt_idx)
            if valid:
                gt_missed_valid.append(gt_idx)

        if matched:
            gt_deviation = np.mean(deviations)
            all_deviation.append(gt_deviation) 
            all_covered.append(covered)
            all_dura.append(gt_dura)

    if debug:
        # if dt_false_pos_all:
        #     for dt_fp_idx in dt_false_pos_all:
        #         rprint(f'img_id: {img_id}, dt_id:{dt_fp_idx}, fp_num: {len(dt_false_pos_all)}')
        #         rprint(f'dt: {dts[dt_fp_idx]}')
        #         # rprint(f'gt: {gts[dt_fp_idx]}')
        pass
                
    res = {
        # TODO TP and FN are calculated based on dt and gt respectively
        'dt_false_pos_all': len(dt_false_pos_all),
        'dt_true_pos_all': len(dt_true_pos_all),
        'gt_matched_all': len(gt_matched_all),
        'gt_missed_all': len(gt_missed_all),
        'dt_true_pos_valid': len(dt_true_pos_valid),
        'gt_matched_valid': len(gt_matched_valid),
        'gt_missed_valid': len(gt_missed_valid),
        'all_deviation': all_deviation,
        'all_covered': all_covered,
        'all_dura': all_dura
    }
    return res


def accumulate_wistle_results(img_to_whistles,valid_gt=False, debug=False):
    """accumulate the whistle results for all images (segment or entire audio)"""
    accumulated_res = {
        'dt_false_pos_all': 0,
        'dt_true_pos_all': 0,
        'gt_matched_all': 0,
        'gt_missed_all': 0,
        'dt_true_pos_valid': 0,
        'gt_matched_valid': 0,
        'gt_missed_valid': 0,
        'all_deviation': [],
        'all_covered': [],
        'all_dura': []
    }
    for img_id, whistles in img_to_whistles.items():
        res = compare_whistles(**whistles, valid_gt = valid_gt, debug=debug)
        # rprint(f'img_id: {img_id}')
        # rprint(sumerize_whisle_results(res))
        accumulated_res['dt_false_pos_all'] += res['dt_false_pos_all']
        accumulated_res['dt_true_pos_all'] += res['dt_true_pos_all']
        accumulated_res['dt_true_pos_valid'] += res['dt_true_pos_valid']
        accumulated_res['gt_matched_all'] += res['gt_matched_all']
        accumulated_res['gt_matched_valid'] += res['gt_matched_valid']
        accumulated_res['gt_missed_all'] += res['gt_missed_all']
        accumulated_res['gt_missed_valid'] += res['gt_missed_valid']
        accumulated_res['all_deviation'].extend(res['all_deviation'])
        accumulated_res['all_covered'].extend(res['all_covered'])
        accumulated_res['all_dura'].extend(res['all_dura'])
    return accumulated_res

def sumerize_whisle_results(accumulated_res):
    """sumerize the whistle results"""
    accumulated_res = copy.deepcopy(accumulated_res)
    dt_fp = accumulated_res['dt_false_pos_all']
    dt_tp = accumulated_res['dt_true_pos_all']
    dt_tp_valid = accumulated_res['dt_true_pos_valid']
    gt_tp = accumulated_res['gt_matched_all']
    gt_tp_valid = accumulated_res['gt_matched_valid']
    gt_fn = accumulated_res['gt_missed_all']
    gt_fn_valid = accumulated_res['gt_missed_valid']

    precision = dt_tp / (dt_tp + dt_fp) if (dt_tp + dt_fp) > 0 else 0
    precision_valid = dt_tp_valid / (dt_tp_valid + dt_fp) if (dt_tp_valid + dt_fp) > 0 else 0
    recall = gt_tp / (gt_tp + gt_fn) if (gt_tp + gt_fn) > 0 else 0
    recall_valid = gt_tp_valid / (gt_tp_valid + gt_fn_valid) if (gt_tp_valid + gt_fn_valid) > 0 else 0
    frag = dt_tp / gt_tp if gt_tp > 0 else 0
    frag_valid = dt_tp_valid / gt_tp_valid if gt_tp_valid > 0 else 0

    accumulated_res['all_deviation'] = np.mean(accumulated_res['all_deviation']).item()
    accumulated_res['all_covered'] = np.sum(accumulated_res['all_covered']).item()
    accumulated_res['all_dura'] = np.sum(accumulated_res['all_dura']).item()
    coverage = accumulated_res['all_covered'] / accumulated_res['all_dura'] if accumulated_res['all_dura'] > 0 else 0

    summary = {
        'precision': precision,
        'recall': recall,
        'frag': frag,
        'coverage': coverage,
        'gt_n':(gt_tp_valid + gt_fn_valid),
        'dt_n':(dt_tp_valid + dt_fp),
        'precision_valid': precision_valid,
        'recall_valid': recall_valid,
        'frag_valid': frag_valid,
    }
    return summary