# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass
import datetime
import itertools
import json
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
from whistle_prompter import utils
import os

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


    def results2whistles(self, results: Sequence[dict],image_ids,
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
        
        results_dict = defaultdict(dict)

        w_id = 0
        for idx, result in enumerate(tqdm(results, desc='Convert dt to whistle')):
            image_id = result.get('img_id', idx)
            if image_id not in image_ids:
                continue
            labels = result['labels']
            scores = result['scores']

            # segm results
            masks = result['masks']
            mask_scores = result.get('mask_scores', scores)
            for i, label in enumerate(labels):
                if float(mask_scores[i]) <  filter_dt:
                    continue
                if isinstance(masks[i]['counts'], bytes):
                    masks[i]['counts'] = masks[i]['counts'].decode()
                bitmap = maskUtils.decode(masks[i])
                if bitmap.sum() > 0:
                    whistle= mask_to_whistle(bitmap)
                    for w in whistle:
                        results_dict[image_id][w_id] = w
                        w_id += 1


        # TODO: dump whistles

        # result_files['bbox'] = f'{outfile_prefix}.bbox.json'
        # result_files['proposal'] = f'{outfile_prefix}.bbox.json'
        # dump(bbox_json_results, result_files['bbox'])

        # if segm_json_results is not None:
        #     result_files['segm'] = f'{outfile_prefix}.segm.json'
        #     dump(segm_json_results, result_files['segm'])

        return results_dict

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

        stems = json.load(open('../data/cross/meta.json'))
        stems = stems['test']

        # DEBUG: 
        # stems = ['palmyra092007FS192-070924-205305']
        stems = ['Qx-Tt-SCI0608-N1-060814-123433']

        # add img_id in same audio file
        audio_to_img = defaultdict(dict)

        for id, img in self._coco_api.imgs.items():
            audio_to_img[img['audio_filename']][img['start_frame']] = id  # {stem: {start_frame: img_id}}
        
 
        img_to_whistles = dict()
        for stem in stems:
            start_frames = list(audio_to_img[stem].keys())
            image_ids=  list(audio_to_img[stem].values())
            ordered_idx = np.argsort(start_frames)
            sorted_frames = [start_frames[i] for i in ordered_idx]
            img_ids = [image_ids[i] for i in ordered_idx]

            # dt eval 
            img_to_frames = {img_id: start_frame for img_id, start_frame in zip(image_ids, sorted_frames)} 
            filter_dt = 0.8
            result_whistle = self.results2whistles(preds, img_ids, filter_dt=filter_dt)  # {img_id: {id: whistle ...} }
            coco_whistles = {img_to_frames[img_id]: whistles for img_id, whistles in result_whistle.items()}  # {start_frame: {id: whistle} ...}

            # # gt eval
            # coco_whistles = dict()
            # for i, (sf, img_id) in enumerate(zip(sorted_frames, img_ids)):
            #     ann_ids = self._coco_api.getAnnIds(imgIds=img_id)
            #     anns = self._coco_api.loadAnns(ann_ids)
            #     coco_whistles[sf] = {ann['id']: mask_to_whistle(self._coco_api.annToMask(ann)) for ann in anns}  # {start_frame: {ann_id: whistle} ...}
            


            # Merge whistles that are cut at frame boundaries
            merged_whistles = []
            dt_whistles = []  # Will contain both merged and non-merged whistles
            frame_width = 1500  # 1500 pixels = 300ms (since 1 pixel = 0.2ms)
            freq_height = 769  # 769 pixels = 125Hz (since 1 pixel = 125Hz)
            top_cutoff = 368

            sorted_frames = sorted(coco_whistles.keys())
            # Track whistles that have already been merged to avoid re-merging
            merged_ids = set()

            # First pass: identify and merge whistles at frame boundaries
            for i in range(len(sorted_frames) - 1):
                current_frame = sorted_frames[i]
                next_frame = sorted_frames[i + 1]
                
                # Check if frames are consecutive (300ms apart)
                if next_frame - current_frame == frame_width:
                    current_whistles = coco_whistles[current_frame]
                    next_whistles = coco_whistles[next_frame]
                    
                    # Find whistles that might be cut at right edge of current frame
                    for c_id, c_whistle in current_whistles.items():
                        # Skip if this whistle has already been merged
                        if c_id in merged_ids:
                            continue
                            
                        # Check if whistle is cut at right edge (x values close to frame width)
                        if np.any(c_whistle[:, 0] >= frame_width - 1):  # Within 1ms of edge
                            # Look for matching whistles in next frame cut at left edge
                            for n_id, n_whistle in next_whistles.items():
                                # Skip if this whistle has already been merged
                                if n_id in merged_ids:
                                    continue
                                    
                                # Check if whistle starts at left edge
                                if np.any(n_whistle[:, 0] < 1):  # Within 1ms of edge
                                    # Get rightmost points of current whistle
                                    c_right_points = c_whistle[c_whistle[:, 0] >= frame_width - 1]
                                    # Get leftmost points of next whistle
                                    n_left_points = n_whistle[n_whistle[:, 0] < 1]
                                    
                                    # Check if y-coordinates are similar (frequency alignment within 250Hz)
                                    # Since each pixel is 125Hz, a difference of 2 pixels = 250Hz
                                    c_avg_y = np.mean(c_right_points[:, 1])
                                    n_avg_y = np.mean(n_left_points[:, 1])
                                    
                                    if abs(c_avg_y - n_avg_y) <= 2:  # 2 pixels = 250Hz threshold
                                        # Merge the whistles
                                        # Adjust x-coordinates of next whistle by adding frame width (300ms)
                                        n_whistle_adjusted = n_whistle.copy()
                                        n_whistle_adjusted[:, 0] += frame_width
                                        
                                        # Combine the whistles
                                        merged_whistle = np.vstack([c_whistle, n_whistle_adjusted])
                                        # Sort by time (x-coordinate)
                                        merged_whistle = merged_whistle[merged_whistle[:, 0].argsort()]
                                        merged_whistle[:, 0] += current_frame 
                                        
                                        # Convert to time-frequency coordinates
                                        merged_whistle_pix = merged_whistle.copy().astype(np.float32)
                                        # merged_whistle_pix[:, 0] =   (merged_whistle_pix[:, 0]-0.5) * 0.002 # Convert to milliseconds
                                        # merged_whistle_pix[:, 1] = (freq_height - 1 - merged_whistle_pix[:, 1] - 0.5) * 125  # Convert to frequency
                                        
                                        # Store the merged whistle
                                        merged_whistles.append(merged_whistle_pix)
                                        
                                        # Mark these whistles as merged so they won't be considered again
                                        merged_ids.add(c_id)
                                        merged_ids.add(n_id)

            # Second pass: add all non-merged whistles to the complete list
            for frame, whistle_dict in coco_whistles.items():
                for whistle_id, whistle in whistle_dict.items():
                    # Skip if this whistle was already part of a merge
                    if whistle_id in merged_ids:
                        continue
                    
                    # Convert to time-frequency coordinates
                    whistle_pix = whistle.copy().astype(np.float32)
                    whistle_pix[:, 0] = frame + whistle_pix[:, 0]  # Add frame start
                    
                    # Add to the complete list
                    dt_whistles.append(whistle_pix)

            # Add merged whistles to the complete list
            dt_whistles.extend(merged_whistles)
            
            # Filter out whistle segments that are outside the frequency range
            filtered_dt_whistles = []
            for whistle in dt_whistles:
                # Find points that are below the top_cutoff (points below top_cutoff are valid)
                valid_indices = whistle[:, 1] > top_cutoff
                
                # If no valid points, skip this whistle entirely
                if not np.any(valid_indices):
                    continue
                
                # Find continuous segments above the cutoff
                segments = []
                current_segment = []
                for i, (point, is_valid) in enumerate(zip(whistle, valid_indices)):
                    if is_valid:
                        current_segment.append(point)
                    elif current_segment:  # End of a valid segment
                        if len(current_segment) > 10:  # remove too short segments
                            segments.append(np.array(current_segment))
                        current_segment = []
                
                # Don't forget the last segment if it exists
                if current_segment and len(current_segment) > 10:
                    segments.append(np.array(current_segment))
                
                # Add all valid segments to our filtered list
                filtered_dt_whistles.extend(segments)
            dt_whistles = filtered_dt_whistles


            

            # save the whistles to binary file
            dt_whistles_tf = [pix_to_tf(whistle, height=freq_height) for whistle in dt_whistles]
            tonnal_save(dt_whistles_tf, stem)

            binfile = os.path.join('../data/cross/anno', f'{stem}.bin')
            gt_tonals = utils.load_annotation(binfile)

            def unique_pix(traj):
                unique_x = np.unique(traj[:, 0])
                averaged_y = np.zeros_like(unique_x)
                for i, x in enumerate(unique_x):
                    y_values = traj[traj[:, 0] == x][:, 1]
                    averaged_y[i] = int(np.round(np.mean(y_values)))
                unique_traj = np.column_stack((unique_x, averaged_y))
                return unique_traj

            gt_tonals_pix = []
            for i, gt_traj in enumerate(gt_tonals):
                traj_pix = utils.tf_to_pix(gt_traj, width=np.inf)
                traj_pix = unique_pix(traj_pix)
                gt_tonals_pix.append(traj_pix)

            img_to_whistles[stem]={
                'gts': gt_tonals_pix,
                'dts': dt_whistles,
                'w': torch.inf,
                'img_id': stem,
            }
        sum_gts = sum([len(whistles['gts']) for whistles in img_to_whistles.values()])
        sum_dts = sum([len(whistles['dts']) for whistles in img_to_whistles.values()])
        rprint(f'gathered {sum_gts} gt whistles, {sum_dts} dt whistles within')
        eval_results = OrderedDict()

        res = accumulate_wistle_results(img_to_whistles, valid_gt=True, valid_len = 75, deviation_tolerence= 350/125)
        summary = summarize_whistle_results(res)
        rprint(summary)

        return eval_results

def pix_to_tf(pix, height):
    """Convert pixel coordinates to time-frequency coordinates."""
    time = (pix[:, 0] + 0.5) * 0.002  # Convert to seconds
    freq = (height - 1 - pix[:, 1] - 0.5) * 125  # Convert to frequency in Hz
    return np.column_stack((time, freq))

@dataclass
class contour:
    time: float
    freq: float

def tonnal_save(tonnals, stem, model_name = 'mask2former'):
    """Save the tonnals to a silbido binary file
    Args:
        tonnals: list of tonnals array
        preprcoess to list of dictionaries of tonnals in format
        {"tfnodes": [
                {"time": 3.25, "freq": 50.125, "snr": 6.6, "phase": 0.25, "ridge": 1.0},
                {"time":...},
                ...,]
        }
    """
    # convert to dataclass
    tonnals = [contour(time=tonnal[:, 0], freq=tonnal[:, 1]) for tonnal in tonnals]

    from whistle_prompter.utils.write_bin import writeTimeFrequencyBinary
    writeTimeFrequencyBinary(f'outputs/{stem}_{model_name}_dt.bin', tonnals)


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

def mask_to_whistle(mask, method='bresenham', max_gap=25, min_segment_ratio = 0.1):
    """convert the instance mask to whistle contour, use skeleton methods
    
    Args
        mask: instance mask (H, W)
    Return
        whistle: (N,2) in pixel coordinates
    """
    mask = mask.astype(np.uint8)
    skeleton = skeletonize(mask).astype(np.uint8)
    border_mask = np.zeros_like(mask, dtype=bool)
    border_mask[:, [0, -1]] = True
    border_pixels = mask & border_mask
    skeleton = skeleton | border_pixels
    whistle = np.array(np.nonzero(skeleton)).T # [(y, x]
    whistle = np.flip(whistle, axis=1)  # [(x, y)]
    whistle = np.unique(whistle, axis=0)  # remove duplicate points
    whistle = whistle[whistle[:, 0].argsort()]
    assert whistle.ndim ==2 and whistle.shape[1] == 2, f"whistle shape: {whistle.shape}"

    # Split into segments based on point distances
    segments = []
    current_segment = [whistle[0]]
    skip_pre = False

    for i in range(1, len(whistle)):
        current_point = whistle[i]
        if not skip_pre:
            prev_point = whistle[i-1]
        else:  
            skip_pre = False

        if abs(current_point[0] - prev_point[0]) < max_gap:
            if abs(current_point[1] - prev_point[1]) < max_gap:
                current_segment.append(current_point)
            else: # skip outliers
                skip_pre = True
                continue  # Skip harmonics that are too far apart
        else:
            segments.append(np.array(current_segment))
            current_segment = [current_point]
    segments.append(np.array(current_segment))
    
    # If no segments met the length requirement, return empty array
    if not segments:
        return np.zeros((0, 2), dtype=np.int32)
    
    # Sort segments by their x-range length (descending)
    segments.sort(key=lambda s: s[-1, 0] - s[0, 0], reverse=True)
    
      # Determine which segments to keep (long enough compared to main segment)
    main_segment_length = segments[0][-1, 0] - segments[0][0, 0]
    min_length = main_segment_length * min_segment_ratio
    
    # Process all segments that meet the length requirement
    result_whistles = []
    for segment in segments:
        segment_length = segment[-1, 0] - segment[0, 0]
        if segment_length >= min_length:
            # Connect points within this segment
            segment_points = segment.tolist()
            processed_segment = [segment_points[0]]
            
            for i in range(len(segment_points) - 1):
                current = segment_points[i]
                next_point = segment_points[i + 1]
                
                # Generate intermediate points
                if method == 'bresenham':
                    intermediate_points = bresenham_line(current, next_point)
                elif method == 'midpoint':
                    intermediate_points = midpoint_interpolation(current, next_point)
                elif method == 'bezier':
                    intermediate_points = bezier_grid_path(segment_points, i, i+1)
                elif method == 'weighted':
                    intermediate_points = simple_weighted_path(segment_points, i)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                # Add intermediate points
                processed_segment.extend(intermediate_points[1:])
            
            # Convert to numpy array and process x-coordinate grouping
            processed_segment = np.array(processed_segment)
            unique_x = np.unique(processed_segment[:, 0])
            averaged_y = np.zeros_like(unique_x)
            for i, x in enumerate(unique_x):
                y_values = processed_segment[processed_segment[:, 0] == x][:, 1]
                averaged_y[i] = int(np.round(np.mean(y_values)))
            unique_whistle = np.column_stack((unique_x, averaged_y))
            
            result_whistles.append(unique_whistle)

    return result_whistles

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
        
        nested_gt_whistles = [mask_to_whistle(mask) for mask in gt_masks]
        gt_whistles = [w for whistles in nested_gt_whistles for w in whistles]
        
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

        nested_dt_whistles = [mask_to_whistle(mask) for mask in dt_masks if mask.sum()>0]
        dt_whistles = [w for whistles in nested_dt_whistles for w in whistles]
        
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


def compare_whistles(gts, dts, w, img_id, boudns_gt=None, valid_gt = False, valid_len = 75, deviation_tolerence = 350/125, debug=False):
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

    if type(valid_len) == int:
        delt= 1
    else:
        delt = 0.002

    for gt_idx, gt in enumerate(gts):
        gt_start_x = max(0, gt[:, 0].min())
        gt_end_x = min(w -delt , gt[:, 0].max())
        gt_dura = gt_end_x + delt - gt_start_x  # add 1 in pixel
        gt_durations[gt_idx] = gt_dura
        gt_ranges[gt_idx] = (gt_start_x, gt_end_x)
  
    for dt_idx, dt in enumerate(dts):
        dt_start_x = max(0, dt[:, 0].min())
        dt_end_x = min(w, dt[:, 0].max())
        dt_ranges[dt_idx] = (dt_start_x, dt_end_x)
        dt_durations[dt_idx] = dt_end_x + delt - dt_start_x # add 1 in pixel
    
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
            if gt_dura < valid_len: # or boudns_gt[gt_idx] < 3:
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
            deviation_tolerence = deviation_tolerence
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
                
                covered += dt_ovlp_xs.max() - dt_ovlp_xs.min() + delt
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


def accumulate_wistle_results(img_to_whistles, valid_gt, valid_len=75,deviation_tolerence = 350/125, debug=False):
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
        res = compare_whistles(**whistles, valid_gt = valid_gt, valid_len = valid_len, deviation_tolerence = deviation_tolerence,   debug=debug)
        rprint(f'img_id: {img_id}')
        rprint(summarize_whistle_results(res))
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

def summarize_whistle_results(accumulated_res):
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
        'gt_all': gt_tp + gt_fn,
        'dt_all': dt_tp + dt_fp,
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