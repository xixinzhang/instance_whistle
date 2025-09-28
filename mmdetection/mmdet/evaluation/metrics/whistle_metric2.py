# Copyright (c) OpenMMLab. All rights reserved.
from dataclasses import dataclass
import datetime
import itertools
import cv2
import  yaml
import json
import os.path as osp
import tempfile
from collections import OrderedDict, defaultdict, deque
from time import time
from typing import Dict, List, Optional, Sequence, Tuple, Union
from rich import print as rprint
from skimage.morphology import skeletonize
from copy import deepcopy
from scipy.spatial.distance import cdist

import numpy as np
from scipy.signal import medfilt2d
import librosa
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

import pycocotools.mask as maskUtils
from rich import print as rprint
from whistle_prompter import utils
from whistle_prompter.utils.write_binary import writeTimeFrequencyBinary, writeContoursBinary
import os
# FIRST convert warnings to exceptions
import warnings
from numpy.exceptions import RankWarning
warnings.filterwarnings('error', category=RankWarning)

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
        stems = yaml.safe_load(open('../data/cross/meta.yaml'))
        stems = stems['test']

        # DEBUG/VAl: 
        stems = ['palmyra092007FS192-070928-040000']


        # add img_id in same audio file
        audio_to_img = defaultdict(dict)

        for id, img in self._coco_api.imgs.items():
            audio_to_img[img['audio_filename']][img['start_frame']] = id  # {stem: {start_frame: img_id}}
        

        block_size = 1500  # 1500 pixels = 300ms (since 1 pixel = 0.2ms)
        freq_height = 769  # 769 pixels = 125Hz (since 1 pixel = 125Hz)

        # post processing
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

            # Merge whistles that are cut at frame boundaries
            dt_whistles = connect_whistles_at_boundaries(coco_whistles, block_size)
            
            # Filter out whistle segments that are outside the frequency range
            dt_whistles = cut_whistle_outrange(dt_whistles)
            # Filter out too short whistle segments
            dt_whistles = filter_whistles(dt_whistles)

            # apply NMS to remove overlapping whistles
            dt_whistles = whistle_nms(dt_whistles)

            # Merge groups of segments after NMS
            dt_whistles = merge_whistle_groups(dt_whistles)

            # save the whistles to binary file
            waveform, sample_rate = load_wave_file(f'../data/cross/audio/{stem}.wav')
            spect_power_db= wave_to_spect(waveform, sample_rate)
            H, W = spect_power_db.shape[-2:]
            # clip whistles to [0, H-1][0, W-1]
            dt_whistles = [np.clip(whistle, [0, 0], [W-1, H-1]) for whistle in dt_whistles]

            spect_snr = np.zeros_like(spect_power_db)
            broadband = 0.01
            for i in range(0, W, block_size):
                spect_snr[:, i:i+block_size] = snr_spect(spect_power_db[:, i:i+block_size], click_thr_db=10, broadband_thr_n=broadband*H )
            spect_snr = np.flipud(spect_snr) # flip frequency axis, low freq at the bottom
            tonals_snr = [spect_snr[w[:, 1].astype(int),w[:, 0].astype(int)] for w in dt_whistles]
            dt_whistles_tf = [pix_to_tf(whistle, height=freq_height) for whistle in dt_whistles]
            # tonal_save(stem, dt_whistles_tf, tonals_snr, model_name='mask2former')

            def unique_pix(traj):
                unique_x = np.unique(traj[:, 0])
                averaged_y = np.zeros_like(unique_x)
                for i, x in enumerate(unique_x):
                    y_values = traj[traj[:, 0] == x][:, 1]
                    averaged_y[i] = int(np.round(np.mean(y_values)))
                unique_traj = np.column_stack((unique_x, averaged_y))
                return unique_traj
            
            binfile = os.path.join('../data/cross/anno_refined', f'{stem}.bin')
            gt_tonals = utils.load_tonal_reader(binfile)

            gt_whistles = []
            for i, gt_traj in enumerate(gt_tonals):
                traj_pix = utils.tf_to_pix(gt_traj, width=np.inf)
                traj_pix = unique_pix(traj_pix)
                gt_whistles.append(traj_pix)

            gt_whistles = cut_whistle_outrange(gt_whistles)
            gt_whistles = filter_whistles(gt_whistles)

            img_to_whistles[stem]={
                'gts': gt_whistles,
                'dts': dt_whistles,
                'w': torch.inf,
                'img_id': stem,
            }

        sum_gts = sum([len(whistles['gts']) for whistles in img_to_whistles.values()])
        sum_dts = sum([len(whistles['dts']) for whistles in img_to_whistles.values()])
        rprint(f'gathered {sum_gts} gt whistles, {sum_dts} dt whistles within')
        eval_results = OrderedDict()

        res = accumulate_whistle_results(img_to_whistles, debug=True, valid_gt=True, valid_len = 75, deviation_tolerence= 350/125)
        summary = summarize_whistle_results(res)
        rprint(summary)

        return eval_results


def connect_whistles_at_boundaries(coco_whistles, block_size=1500, freq_threshold=2):
    """
    Merge whistles that are cut at frame boundaries.

    Args:
        coco_whistles: {start_frame: {id: whistle}}
        block_size: int, frame width in pixels
        freq_threshold: int, max allowed y difference (in pixels) for merging

    Returns:
        merged_whistles: list of merged whistle arrays
        merged_ids: set of whistle ids that were merged
    """
    merged_whistles = []
    merged_ids = set()
    sorted_frames = sorted(coco_whistles.keys())

    # First pass: identify and merge whistles at frame boundaries
    for i in range(len(sorted_frames) - 1):
        current_frame = sorted_frames[i]
        next_frame = sorted_frames[i + 1]
        if next_frame - current_frame != block_size:
            continue
        current_whistles = coco_whistles[current_frame]
        next_whistles = coco_whistles[next_frame]
        for c_id, c_whistle in current_whistles.items():
            if c_id in merged_ids or not np.any(c_whistle[:, 0] >= block_size - 1):
                continue
            for n_id, n_whistle in next_whistles.items():
                if n_id in merged_ids or not np.any(n_whistle[:, 0] < 1):
                    continue
                c_right_points = c_whistle[c_whistle[:, 0] >= block_size - 1]
                n_left_points = n_whistle[n_whistle[:, 0] < 1]
                c_avg_y = np.mean(c_right_points[:, 1])
                n_avg_y = np.mean(n_left_points[:, 1])
                if abs(c_avg_y - n_avg_y) <= freq_threshold:
                    n_whistle_adjusted = n_whistle.copy()
                    n_whistle_adjusted[:, 0] += block_size
                    merged_whistle = np.vstack([c_whistle, n_whistle_adjusted])
                    merged_whistle = merged_whistle[merged_whistle[:, 0].argsort()]
                    merged_whistle[:, 0] += current_frame
                    merged_whistles.append(merged_whistle.astype(int))
                    merged_ids.add(c_id)
                    merged_ids.add(n_id)

    # Second pass: add all non-merged whistles
    dt_whistles = []
    for frame, whistle_dict in coco_whistles.items():
        for whistle_id, whistle in whistle_dict.items():
            if whistle_id in merged_ids:
                continue
            whistle_pix = whistle.copy().astype(int)
            whistle_pix[:, 0] = frame + whistle_pix[:, 0]
            dt_whistles.append(whistle_pix)

    # Add merged whistles to the complete list
    dt_whistles.extend(merged_whistles)
    return dt_whistles

def cut_whistle_outrange(whistles, 
                        top_cutoff = 368, # (96000-50000)/125
                        bottom_cutoff = 729 # 769 - 5000/125
            ):
    """Cut whistles that are out of range."""
    filtered_whistles = []
    for whistle in whistles:
        # Find points that are below the top_cutoff (points below top_cutoff are valid)
        try:
            valid_indices = (whistle[:, 1] > top_cutoff) & (whistle[:, 1] < bottom_cutoff)
        except:
            import pdb; pdb.set_trace()
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
                segments.append(np.array(current_segment))
                current_segment = []
                    
        # the last segment if it exists
        if current_segment:
            segments.append(np.array(current_segment))
        
        # Add all valid segments to our filtered list
        filtered_whistles.extend(segments)
    return filtered_whistles

def filter_whistles(whistles, length_thre=10):
    """Filter whistles based on length and frequency range."""
    filtered_whistles = []
    for whistle in whistles:
        freq_range = whistle[-1].max() - whistle[0].min()
        # Check if the whistle has enough unique x-coordinates
        if len(whistle) < length_thre or freq_range < length_thre:
            continue
        else:
            filtered_whistles.append(whistle)
    return filtered_whistles

 # Merge groups of whistle segments using polynomial fit
def merge_whistle_groups(whistles, min_fit_window = 20, max_dis=25, r2_threshold=0.92, max_degree=2, deviation_threshold=2):
    if len(whistles) == 0:
        return []
    sorted_whistles = sorted(whistles, key=lambda w: w[0, 0])
    merged = []
    used = set()
    i = 0
    current = None
    while i < len(sorted_whistles):
        if i in used:
            i += 1
            continue
        if current is None: 
            current = sorted_whistles[i].copy()
        candidates = []
        # Find all candidate segments within max_gap_x after current
        for j in range(i + 1, len(sorted_whistles)):
            if j in used:
                continue
            next_seg = sorted_whistles[j]
            gap_x = next_seg[0, 0] - current[-1, 0]
            gap_y = np.abs(next_seg[0, 1] - current[-1, 1])
            overlap_start = max(current[0, 0], next_seg[0, 0])
            overlap_end = min(current[-1, 0], next_seg[-1, 0])
            # Case 1: Not overlapped in x, but within gap
            if gap_x >= 0 and np.linalg.norm([gap_x, gap_y]) <= max_dis:
                # Try direct connection (endpoint to startpoint)
                # Evaluate fitness of connection using polynomial fit
                fit_window = min(min_fit_window, len(current), len(next_seg))
                x_fit = np.concatenate([current[-fit_window:,0], next_seg[:fit_window,0]])
                y_fit = np.concatenate([current[-fit_window:,1], next_seg[:fit_window,1]])
                try:
                    poly = PolynomialRegression(x_fit, y_fit, degree=max_degree)
                except:
                    import pdb; pdb.set_trace()
                x_connect = np.arange(current[-1,0] + 1, next_seg[0,0])
                # Only add as candidate if fit is good enough
                candidates.append({
                    'type': 'connect',
                    'idx': j,
                    'R2': poly.R2,
                    'next_seg': next_seg if gap_x > 0 else next_seg[1:],
                    'connection': np.column_stack((x_connect, poly.predict(x_connect))).astype(int)
                })

            # Case 2: Overlapped in x
            if overlap_end > overlap_start:
                mask_cur = (current[:, 0] >= overlap_start) & (current[:, 0] <= overlap_end)
                mask_next = (next_seg[:, 0] >= overlap_start) & (next_seg[:, 0] <= overlap_end)
                cur_overlap = current[mask_cur]
                next_overlap = next_seg[mask_next]
                if len(cur_overlap) > 0 and len(next_overlap) > 0:
                    # Only consider merging if mean y deviation in overlap is small
                    deviation = np.mean(np.abs(cur_overlap[:, 1] - next_overlap[:, 1]))
                    if deviation <= deviation_threshold:
                        x_connect = np.arange(overlap_start, overlap_end+1)
                        connection = np.column_stack((x_connect, (cur_overlap[:, 1] + next_overlap[:, 1]) / 2))
                        # Overlap coincides in x and y, evaluate fitness to extend segment 1 with segment 2's non-overlapped part
                        fit_window = min(min_fit_window, len(current[:overlap_start]), len(next_overlap[overlap_end:]))
                        x_fit = np.concatenate([current[:overlap_start, 0], connection[:, 0], next_overlap[overlap_end:, 0]])
                        y_fit = np.concatenate([current[:overlap_start, 1], connection[:, 1], next_overlap[overlap_end:, 1]])
                        try:
                            poly = PolynomialRegression(x_fit, y_fit, degree=max_degree)
                        except:
                            import pdb; pdb.set_trace()
                        candidates.append({
                            'type': 'overlap',
                            'idx': j,
                            'R2': poly.R2,
                            'non_overlap_next': (next_seg[next_seg[:, 0] > overlap_end]).astype(int)
                        })

        # Select best candidate to connect/merge
        if candidates:
            # Prioritize by R2 (desc)
            candidates.sort(key=lambda d: (-d['R2']))
            best = candidates[0]
            if best['type'] == 'connect' and best['R2'] > r2_threshold:
                # Connect segments
                current = np.vstack([current, best['connection'], best['next_seg']])
                used.add(best['idx'])
                continue
            elif best['type'] == 'overlap' and best['R2'] > r2_threshold:
                # Merge by extending with non-overlapping part
                if len(best['non_overlap_next']) > 0:
                    current = np.vstack([current, best['non_overlap_next']])
                used.add(best['idx'])
                continue
        merged.append(current)
        i += 1
        current = None

    return merged

def whistle_nms(dt_whistles: List[np.ndarray], 
                         freq_deviation_threshold: float = 2,
                         overlap_threshold: float = 0.7) -> List[int]:
    """
    Ultra-fast NMS using frequency deviation over overlapped time ranges.
    Much more efficient than point-by-point matching.
    
    Args:
        dt_whistles: List of nx2 arrays (time, freq) coordinates (sorted by time)
        freq_deviation_threshold: Max mean frequency deviation for overlapped regions
        overlap_threshold: Minimum time overlap ratio to consider suppression
    
    Returns:
        List of indices of whistles to keep
    """
    if len(dt_whistles) == 0:
        return []
    
    # Pre-process: ensure time-sorted and extract time ranges
    processed_whistles = []
    lengths = []
    suppressed = set()

    for idx, whistle in enumerate(dt_whistles):
        if len(whistle) == 0:
            processed_whistles.append(None)
            lengths.append(0)
            continue

        # Sort by time if needed
        if len(whistle) > 1 and not np.all(whistle[:-1, 0] <= whistle[1:, 0]):
            whistle = whistle[np.argsort(whistle[:, 0])]
        
        time_range = (whistle[0, 0], whistle[-1, 0])
        processed_whistles.append({
            'data': whistle,
            'time_range': time_range
        })
        lengths.append(len(whistle))
    sorted_indices = np.argsort(lengths)[::-1]

    keep = []
    
    for i, idx in enumerate(sorted_indices):
        if idx in suppressed:
            continue
            
        keep.append(idx)
        current_whistle = processed_whistles[idx]
        
        if current_whistle is None:
            continue
        
        # Check candidates for suppression using deviation method
        for idx2 in sorted_indices[i + 1:]:
            if idx2 in suppressed:
                continue
                
            candidate_whistle = processed_whistles[idx2]
            if candidate_whistle is None:
                continue
            
            # Check if they should be merged based on overlapped region deviation
            if should_suppress_by_deviation(current_whistle, candidate_whistle, 
                                          freq_deviation_threshold, overlap_threshold):
                suppressed.add(idx2)

    return [dt_whistles[i] for i in keep]


def should_suppress_by_deviation(longer_whistle: dict, shorter_whistle: dict,
                               freq_deviation_threshold: float, overlap_threshold: float) -> bool:
    """
    Check if shorter whistle should be suppressed based on frequency deviation in overlap region.
    """
    # Find overlapped time range
    overlap_start = max(longer_whistle['time_range'][0], shorter_whistle['time_range'][0])
    overlap_end = min(longer_whistle['time_range'][1], shorter_whistle['time_range'][1])
    
    # No overlap
    if overlap_start >= overlap_end:
        return False
    
    # Check if overlap is significant enough
    shorter_duration = shorter_whistle['time_range'][1] - shorter_whistle['time_range'][0]
    if shorter_duration <= 0:
        return False
    
    overlap_duration = overlap_end - overlap_start
    overlap_ratio = overlap_duration / shorter_duration
    
    # Insufficient overlap
    if overlap_ratio < overlap_threshold:
        return False
    
    # Extract overlapped segments and compute deviation
    longer_overlap = extract_time_segment(longer_whistle['data'], overlap_start, overlap_end)
    shorter_overlap = extract_time_segment(shorter_whistle['data'], overlap_start, overlap_end)
    
    if longer_overlap is None or shorter_overlap is None:
        return False
    
    # Calculate frequency deviation between overlapped segments
    try:
        deviation = np.mean(np.abs(longer_overlap - shorter_overlap))
    except Exception as e:
        import pdb; pdb.set_trace()

    return deviation <= freq_deviation_threshold


def extract_time_segment(whistle_data: np.ndarray, start_time: float, end_time: float) -> Optional[np.ndarray]:
    """
    Extract whistle segment within time range.
    """
    if len(whistle_data) == 0:
        return None
    
    # Find points within time range
    mask = (whistle_data[:, 0] >= start_time) & (whistle_data[:, 0] <= end_time)
    segment = whistle_data[mask]
    
    return segment if len(segment) > 0 else None



def wave_to_spect(
        waveform, 
        sample_rate=None,
        frame_ms=8, 
        hop_ms=2, 
        pad=0, 
        n_fft=None, 
        hop_length=None, 
        top_db=None, 
        center = False, 
        amin = 1e-16, 
        **kwargs
    ):
    """Convert waveform to raw spectrogram in power dB scale."""
    # fft params
    if n_fft is None:
        if frame_ms is not None and sample_rate is not None:
            n_fft = int(frame_ms * sample_rate / 1000)
        else:
            raise ValueError("n_fft or frame_ms must be provided.")
    if hop_length is None:
        if hop_ms is not None and sample_rate is not None:
            hop_length = int(hop_ms * sample_rate / 1000)
        else:
            raise ValueError("hop_length or hop_ms must be provided.")

    # spectrogram magnitude
    spect = librosa.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window='hamming',
        center=center,
        pad_mode='reflect',
    )
    # decibel scale spectrogram with cutoff specified by top_db
    spect_power_db = librosa.amplitude_to_db(
        np.abs(spect),
        ref=1.0,
        amin=amin,
        top_db=top_db,
    )
    return spect_power_db # (freq, time)

def load_wave_file(file_path, type='numpy'):
    """Load one wave file."""
    if type == 'numpy':
        waveform, sample_rate = librosa.load(file_path, sr=None)
    elif type == 'tensor':
        import torchaudio
        waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate


def snr_spect(spect_db, click_thr_db, broadband_thr_n):
    meanf_db = np.mean(spect_db, axis=1, keepdims=True)
    click_p = np.sum((spect_db - meanf_db) > click_thr_db, axis=0) > broadband_thr_n
    use_p = ~click_p
    spect_db = medfilt2d(spect_db, kernel_size=[3,3])
    if np.sum(use_p) == 0:
        # Qx-Dc-SC03-TAT09-060516-173000.wav 4500 no use_p
        use_p = np.ones_like(click_p)
    meanf_db = np.mean(spect_db[:, use_p], axis=1, keepdims=True)
    snr_spect_db = spect_db - meanf_db
    return snr_spect_db



def pix_to_tf(pix, height = 769):
    """Convert pixel coordinates to time-frequency coordinates."""
    time = (pix[:, 0]-0.5) * 0.002  # Convert to seconds
    freq = (height - 1 - pix[:, 1] + 0.5) * 125  # Convert to frequency in Hz
    return np.column_stack((time, freq))

@dataclass
class contour:
    time: float
    freq: float

def tonal_save(stem, tonals, tonals_snr=None, model_name = 'mask2former'):
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
    filename = f'outputs/{stem}_{model_name}_dt.bin'
    if tonals_snr is None:
        tonals_ = [contour(time=tonal[:, 0], freq=tonal[:, 1]) for tonal in tonals]
        writeTimeFrequencyBinary(filename, tonals_)
    else:
        tonals_ = [{'tfnodes': [{'time': tf[0], 'freq': tf[1], 'snr': snr} for tf, snr in zip(tfs, snrs)]} for tfs, snrs in zip(tonals, tonals_snr)]
        writeContoursBinary(filename, tonals_, snr=True)


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

def clean_whistle_ori(whistle, max_gap=25, min_segment_ratio = 0.1):
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
                intermediate_points = bresenham_line(current, next_point)
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

def overlay_whistle(whistles, filename, thickness=2):
    """Overlay whistles on the image for visualization.
    Args:
        whistles: List of whistles, each whistle is a list of (x, y) tuples
        filename: Path to the image file
        thickness: Thickness of the lines to draw   
    """
    image = cv2.imread(filename)
    for whistle in whistles:
        color = np.random.randint(0, 255, size=3).tolist()
        for i in range(len(whistle) - 1):
            cv2.line(image, tuple(whistle[i]), tuple(whistle[i + 1]), color, thickness)
    # save the image with overlaid whistles
    out_filename = '/home/asher/Desktop/projects/instance_whistle/tmp.png'
    cv2.imwrite(out_filename, image)

def clean_whistle(whistle, max_gap_x=25, max_gap_y =8, recent_fit = 12):
    """Clean the whistle by removing outliers and smoothing the trajectory.
    Args:
        whistle: sorted list of (x, y) tuples representing the whistle trajectory
    """
    unique_x = np.unique(whistle[:, 0])
    if len(unique_x) < 2:
        if len(whistle) == 1:
            return np.array(whistle)[None, :] 
        else:
            return np.mean(whistle, axis=0, keepdims=True).astype(int)[None, :]

    # check if all x has no neighboring multiple y values
    peaks = OrderedDict()
    for i, x in enumerate(unique_x):
        # y gap
        peaks[x] = []
        y_values = whistle[whistle[:, 0] == x][:, 1]
        # check if y_values are close neighbors or belong to different branches
        if len(y_values) > 1:
            # group neighboring y values, seperate y in different branches
            y_values.sort()
            y_groups = []
            current_group = [y_values[0]]
            dis = np.abs(np.diff(y_values))
            for j in range(1, len(y_values)):
                if dis[j-1] <= 2:
                    current_group.append(y_values[j])
                else:
                    y_groups.append(current_group)
                    current_group = [y_values[j]]
            y_groups.append(current_group)
            # take the mean of each group
            for group in y_groups:
                mean_y = int(np.round(np.mean(group)))
                peaks[x].append(mean_y)
        else:
            peaks[x] = [y_values[0]]
    
    whistles = traverse_peaks(peaks, max_gap_x, max_gap_y, recent_fit)

    return whistles

class PolynomialRegression:
    """
    A class to perform polynomial regression using least squares.
    It calculates the best-fit polynomial coefficients and evaluates the fit.
    """

    def __init__(self, x, y, degree=1):
        """
        Initializes the PolynomialRegression model.

        Args:
            degree (int): The initial desired degree of the polynomial.
                          This will be adjusted based on the number of data points.
        """
        self.degree = degree
        # Coefficients of the polynomial. Stored in decreasing power order (e.g., [a, b, c] for ax^2 + bx + c)
        # to be compatible with numpy.polyval.
        self.coefficients = None
        self.num_observations = 0

        # Evaluation metrics
        self.ssTotal = 0.0
        self.ssRes = 0.0
        self.ssReg = 0.0
        self.R2 = 0.0
        self.R2_ADJ = 0.0
        self.sdRes = 0.0

        self.create_fit(x, y)
        self.evaluate_fit(x, y)

    def get_squared_error(self, x_val, y_actual):
        """
        Calculates the squared error for a given point.

        Args:
            x_val (float): The independent variable value.
            y_actual (float): The actual dependent variable value.

        Returns:
            float: The squared error.
        """
        error = y_actual - self.predict(x_val) # Directly calculate error here
        return error * error

    def predict(self, x_val):
        """
        Predicts the y-value for a given x-value using the calculated polynomial coefficients.
        Leverages numpy.polyval for efficient evaluation.

        Args:
            x_val (float or np.array): The independent variable value(s).

        Returns:
            float or np.array: The predicted dependent variable value(s).
        """
        if self.coefficients is None:
            raise ValueError("Model not fitted yet. Call createFit() first.")

        # numpy.polyval evaluates a polynomial at specific x values.
        # It expects coefficients in decreasing power order (e.g., [a, b, c] for ax^2 + bx + c).
        return np.polyval(self.coefficients, x_val)

    def create_fit(self, x, y):
        """
        Calculates the polynomial coefficients for the best-fit polynomial
        using numpy.polyfit, which is the idiomatic Python way.

        Args:
            x (np.array): Array of independent variable values.
            y (np.array): Array of dependent variable values.
        """
        self.num_observations = len(x)

        if self.num_observations == 1:
            # Handle the special case where we only have one point.
            # We create a 0-order polynomial (a constant).
            self.coefficients = np.array([y[0]]) # Store as a single coefficient for polyval
            self.degree = 0
        else:
            # Adjust the degree based on the number of observations.
            # The degree must be less than the number of points minus 1
            # (for fitting, usually N-1 is max degree for N points).
            self.degree = min(self.num_observations - 2, self.degree)
            if self.degree < 0: # Ensure degree is not negative if num_observations is small (e.g., 1)
                self.degree = 0

            # Use np.polyfit directly for polynomial fitting.
            # np.polyfit returns coefficients in decreasing power order, which is directly
            # compatible with np.polyval.
            try:
                self.coefficients = np.polyfit(x, y, self.degree)
            except np.linalg.LinAlgError as e:
                print(f"Linear algebra error during polynomial fit: {e}")
                self.coefficients = None # Indicate failure to fit

    def evaluate_fit(self, x, y):
        """
        Evaluates the quality of the polynomial fit by calculating various
        statistical metrics.

        Args:
            x (np.array): Array of independent variable values.
            y (np.array): Array of dependent variable values.
        """
        if self.coefficients is None:
            print("Cannot evaluate fit: Model not fitted or fit failed.")
            return

        self.num_observations = len(x) # Ensure num_observations is up-to-date

        # The mean of the dependent variable
        mean_y = np.mean(y)

        # Total Sum of Squares (ssTotal)
        # The sum of all of the squared differences of the dependent variable
        # from the mean.
        self.ssTotal = np.sum((y - mean_y)**2)

        # Residual Sum of Squares (ssRes)
        # Variation caused by the regression model.
        # Sum Of Squared Errors, sum of squares of the residuals.
        y_predicted = self.predict(x) # Use the vectorized predict method
        self.ssRes = np.sum((y - y_predicted)**2)

        # Coefficient of determination (R-squared)
        # R2 measures how well the regression predictions approximate the real data points.
        # R2 = 1 - (SS_res / SS_total)
        self.R2 = 1.0 if self.ssTotal == 0 else 1 - (self.ssRes / self.ssTotal)

        # Adjusted coefficient of determination (Adjusted R-squared)
        # Adjusted R2 accounts for the number of predictors in the model.
        # It is useful for comparing models with different numbers of predictors.
        # Formula: 1 - (1 - R2) * (N - 1) / (N - p - 1)
        # where N is num_observations, p is degree (number of predictors is degree + 1)
        denominator_adj_r2 = self.num_observations - self.degree - 1
        if denominator_adj_r2 > 0:
            self.R2_ADJ = 1 - (1 - self.R2) * (self.num_observations - 1) / denominator_adj_r2
        else:
            self.R2_ADJ = float('nan') # Not well-defined if denominator is zero or negative

        # Regression sum of squares (ssReg)
        # The part of the total variation that is explained by the regression model.
        self.ssReg = self.ssTotal - self.ssRes

        # Standard deviation of residuals (sdRes)
        # A measure of the average distance that the observed values fall from the regression line.
        denominator_sd_res = self.num_observations - self.degree - 1
        if denominator_sd_res > 0:
            self.sdRes = np.sqrt(self.ssRes / denominator_sd_res)
        else:
            self.sdRes = float('nan') # Not well-defined if denominator is zero or negative


def traverse_peaks(peaks: OrderedDict, max_gap_x=25, max_gap_y =8, recent_fit = 12, min_segment_ratio = 0.1):
    # traverse the peaks and extend the whistle
    cleaned_whistles = []
    xs = list(peaks.keys())
    dis = np.abs(np.diff(xs))
    # remove the outliers at the beginning
    indices = np.nonzero(dis < max_gap_x)[0]
    start_idx =  indices[0] if len(indices) > 0 else 0
    for i, (x, ys) in enumerate(peaks.items()):
        if i < start_idx:
            continue
        elif i == start_idx:
            # start the first whistle
            cleaned_whistles.append([(x, ys[0])])
            continue
        else:
            prev_x = list(peaks.keys())[i-1]
            if abs(x - prev_x) > max_gap_x:
                # if the gap is too large, start a new whistle
                cleaned_whistles.append([(x, ys[0])])
                continue
        # add ys to the right branch
        for y in ys:
            min_error = np.inf
            best_fit = 0
            for j, whistle in enumerate(cleaned_whistles):
                # check if the last point of the whistle is close enough to the current point
                if abs(whistle[-1][0] - x) > max_gap_x or abs(whistle[-1][1] - y) > max_gap_y:
                    continue
                # polyfit the whistle
                if len(whistle) > recent_fit:
                    degree = 1
                    x_vector = np.array(whistle)[-recent_fit:, 0]
                    y_vector = np.array(whistle)[-recent_fit:, 1]
                    while True:
                        try:
                            poly = PolynomialRegression(x_vector, y_vector, degree=degree)
                        except RankWarning as e:
                            import pdb; pdb.set_trace()
                        if poly.R2_ADJ > 0.7 or len(x_vector) <= degree*3 or poly.sdRes <= 2:
                            error= poly.get_squared_error(x, y)
                            if error < min_error:
                                min_error = error
                                best_fit = j
                            break
                        degree += 1
                else:
                    for degree in range(1, 3):
                        x_vector = np.array(whistle)[:, 0]
                        y_vector = np.array(whistle)[:, 1]
                        try:
                            poly = PolynomialRegression(x_vector, y_vector, degree=degree)
                        except RankWarning as e:
                            import pdb; pdb.set_trace()
                        if poly.R2_ADJ > 0.7 or len(x_vector) <= degree*3 or poly.sdRes <= 2:
                            error= poly.get_squared_error(x, y)
                            if error < min_error:
                                min_error = error
                                best_fit = j
                            break
            # add the point to the best fit whistle
            if min_error < max_gap_y**2 and x != cleaned_whistles[best_fit][-1][0]:
                cleaned_whistles[best_fit].append((x, y))
            else:
                # if no good fit is found, start a new whistle
                cleaned_whistles.append([(x, y)])

    cleaned_whistles = [np.asarray(whistle) for whistle in cleaned_whistles if len(whistle) > 1]
    # Sort segments by their x-range length (descending)
    cleaned_whistles.sort(key=lambda s: s[-1, 0] - s[0, 0], reverse=True)
    
    result_whistles = []
    if len(cleaned_whistles) == 0:
        return result_whistles
    # Determine which segments to keep (long enough compared to main segment)
    try:
        main_segment_length = cleaned_whistles[0][-1, 0] - cleaned_whistles[0][0, 0]
    except:
        import pdb; pdb.set_trace()
    min_length = main_segment_length * min_segment_ratio

    # fit the gap between the points in each whistle
    for i, whistle in enumerate(cleaned_whistles):
        # check if the whistle has gap in x coordinates
        if len(whistle) < 2:
            continue
        start = whistle[0, 0]
        end = whistle[-1, 0]
        if not np.all(np.isin(np.arange(start, end + 1), whistle[:, 0])):
            processed_segment = [whistle[0]]
            
            for i in range(len(whistle) - 1):
                current = whistle[i]
                next_point = whistle[i + 1]
                # Generate intermediate points
                intermediate_points = bresenham_line(current, next_point)
                # Add intermediate points
                processed_segment.extend(intermediate_points[1:])
        else:
            processed_segment = whistle
        
        processed_segment = np.array(processed_segment)
        unique_x = np.unique(processed_segment[:, 0])
        averaged_y = np.zeros_like(unique_x)
        for i, x in enumerate(unique_x):
            y_values = processed_segment[processed_segment[:, 0] == x][:, 1]
            averaged_y[i] = int(np.round(np.mean(y_values)))
        processed_segment = np.column_stack((unique_x, averaged_y))

        if len(processed_segment) > min_length:
            result_whistles.append(np.array(processed_segment))

    return result_whistles


def mask_to_whistle(mask):
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

    result_whistles = clean_whistle(whistle)

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
    gt_ranges = np.zeros((gt_num, 2))
    gt_durations = np.zeros(gt_num)

    dt_num = len(dts)
    dt_ranges = np.zeros((dt_num, 2))
    dt_durations = np.zeros(dt_num)
    
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
    valid_covered = []
    valid_dura = []

    if valid_gt or debug:
        waveform, sample_rate = load_wave_file(f'../data/cross/audio/{img_id}.wav')
        spect_power_db= wave_to_spect(waveform, sample_rate)
        H, W = spect_power_db.shape[-2:]
        spect_snr = np.zeros_like(spect_power_db)
        block_size = 1500  # 300ms
        broadband = 0.01
        search_row = 4
        ratio_above_snr = 0.3
        for i in range(0, W, block_size):
            spect_snr[:, i:i+block_size] = snr_spect(spect_power_db[:, i:i+block_size], click_thr_db=10, broadband_thr_n=broadband*H)
        spect_snr = np.flipud(spect_snr) # flip frequency axis, low freq at the bottom

    # go through each ground truth
    for gt_idx, gt in enumerate(gts):
        gt_start_x, gt_end_x = gt_ranges[gt_idx]
        gt_dura = gt_durations[gt_idx]
        # Note: remove interpolation and snr validation that filter low gt snr level
        # which requires addition input
        dt_start_xs, dt_end_xs = dt_ranges[:, 0], dt_ranges[:, 1]
        ovlp_cond = (dt_start_xs <= gt_start_x) & (dt_end_xs >= gt_start_x) \
                        | (dt_start_xs>= gt_start_x) & (dt_start_xs <= gt_end_x)
        ovlp_dt_ids = np.nonzero(ovlp_cond)[0]

        matched = False
        deviations = []
        covered = 0
        # Note remove duration filter short < 75 pix whistle
        valid = True
        if valid_gt:
            search_row_low = np.minimum(np.maximum(gt[:, 1] - search_row, 0), H)
            search_row_high = np.maximum(np.minimum(gt[:, 1] + search_row, H), 0)
            gt_cols = gt[:, 0][gt[:, 0] < W]
            try:
                spec_search = [np.max(spect_snr[l:h, col]).item() for i, (l,h, col) in enumerate(zip(search_row_low, search_row_high, gt_cols))] 
            except: 
                import pdb; pdb.set_trace()
            sorted_search_snr = np.sort(spec_search)
            bound_idx = max(0, round(len(sorted_search_snr) * (1- ratio_above_snr))-1)
            gt_snr = sorted_search_snr[bound_idx]
            if gt_dura < valid_len or gt_snr < 3:
                valid= False


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
            if len(deviation)> 0 and np.mean(deviation) <= deviation_tolerence:
                matched = True
                
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
        
        if matched:
            gt_matched_all.append(gt_idx)
            gt_deviation = np.mean(deviations)
            all_deviation.append(gt_deviation)
            all_covered.append(covered)
            if valid:
                gt_matched_valid.append(gt_idx)
                valid_covered.append(covered)
        else:
            gt_missed_all.append(gt_idx)
            if valid:
                gt_missed_valid.append(gt_idx)

        all_dura.append(gt_dura) # move out from matched
        if valid:
            valid_dura.append(gt_dura)

    if debug:
        freq_height = 769
        dt_false_pos_tf_all = [pix_to_tf(dts[idx], height=freq_height) for idx in dt_false_pos_all]
        tonals_snr = [spect_snr[dts[idx][:, 1].astype(int),dts[idx][:, 0].astype(int)] for idx in dt_false_pos_all]
        tonal_save(img_id, dt_false_pos_tf_all, tonals_snr, 'mask2former_swin_fp')
        dt_snrs = [np.mean(snr) for snr in tonals_snr]

        dt_false_neg_tf = [pix_to_tf(gts[idx], height=freq_height) for idx in gt_missed_all]
        tonal_save(img_id, dt_false_neg_tf, model_name='mask2former_swin_fn')

        if len(dt_snrs) > 0:
            # rprint({i+1: dt_snrs[i].item() for i in range(len(dt_snrs))})
            rprint(f'stem: {img_id}, min_snr: {np.min(dt_snrs)}, max_snr: {np.max(dt_snrs)}, mean:{np.mean(dt_snrs)}, above 9: {np.sum(np.array(dt_snrs) > 9)}')
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
        'all_dura': all_dura,
        'valid_covered': valid_covered,
        'valid_dura': valid_dura
    }
    return res


def accumulate_whistle_results(img_to_whistles, valid_gt, valid_len=75,deviation_tolerence = 350/125, debug=False):
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
        'all_dura': [],
        'valid_covered': [],
        'valid_dura': []
    }
    for img_id, whistles in img_to_whistles.items():
        res = compare_whistles(**whistles, valid_gt = valid_gt, valid_len = valid_len, deviation_tolerence = deviation_tolerence, debug=debug)
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
        accumulated_res['valid_covered'].extend(res['valid_covered'])
        accumulated_res['valid_dura'].extend(res['valid_dura'])
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
    accumulated_res['valid_covered'] = np.sum(accumulated_res['valid_covered']).item()
    accumulated_res['valid_dura'] = np.sum(accumulated_res['valid_dura']).item()
    coverage = accumulated_res['all_covered'] / accumulated_res['all_dura'] if accumulated_res['all_dura'] > 0 else 0
    coverage_valid = accumulated_res['valid_covered'] / accumulated_res['valid_dura'] if accumulated_res['valid_dura'] > 0 else 0

    summary = {
        'gt_all': gt_tp + gt_fn,
        'dt_all': dt_tp + dt_fp,
        'precision': precision,
        'recall': recall,
        'f1': 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0,
        'frag': frag,
        'coverage': coverage,
        'gt_n':(gt_tp_valid + gt_fn_valid),
        'dt_n':(dt_tp_valid + dt_fp),
        'precision_valid': precision_valid,
        'recall_valid': recall_valid,
        'f1_valid': 2 * precision_valid * recall_valid / (precision_valid + recall_valid) if (precision_valid + recall_valid) > 0 else 0,
        'frag_valid': frag_valid,
        'coverage_valid': coverage_valid,
    }
    return summary

if __name__ == "__main__":
    import pickle
    whistle = pickle.load(open('/home/asher/Desktop/projects/instance_whistle/test_whistle1.pkl', 'rb'))
    w1 = clean_whistle_ori(whistle)
    w2 = clean_whistle(whistle)