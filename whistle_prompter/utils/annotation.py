import struct
from pathlib import Path
from typing import List

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import clip_by_rect

from .audio import (FRAME_PER_SECOND, FREQ_BIN_RESOLUTION, HOP_MS, N_FRAMES,
                    NUM_FREQ_BINS)


def load_annotation(bin_file: Path) -> list[np.ndarray]:
    """Read the bin file and obtain annotations of each contour
    
    Args:
        bin_file: binary file path that encodes the contour data
    Returns:
        annos: list of contours [(time(s), frequency(Hz))]: [(num_points, 2),...]
    """
    data_format = "dd"  # 2 double-precision [time(s), frequency(Hz)]
    num_dim = 2
    with open(bin_file, "rb") as f:
        bytes = f.read()
        total_bytes_num = len(bytes)
        cur = 0
        annos = []
        if total_bytes_num == 0:
            print(f"{bin_file}: is empty")
            return annos
        while True:
            # get the data length
            num_point = struct.unpack(">i", bytes[cur : cur + 4])[0]
            format_str = f">{num_point * data_format}"
            point_bytes_num = struct.calcsize(format_str)
            cur += 4
            # read the contour data
            data = struct.unpack(f"{format_str}", bytes[cur : cur + point_bytes_num])
            data = np.array(data).reshape(-1, num_dim)
            data = get_dense_annotation(data)  # make the contour continuous
            annos.append(data)
            cur += point_bytes_num
            if cur >= total_bytes_num:
                break
        print(f"Loaded {len(annos)} annotated whistles from {bin_file.stem}.bin")
    return annos  # [(time(s), frequency(Hz)),...]


def get_dense_annotation(traj: np.ndarray, dense_factor: int = 10):
    """Get dense annotation from the trajectory to make it continuous and fill the gaps.

    Args:
        traj: trajectory of the contour  [(time(s), frequency(Hz))]: (num_points, 2)
    """
    time = traj[:, 0]
    sorted_idx = np.argsort(time)
    time = time[sorted_idx]
    freq = traj[:, 1][sorted_idx]
    length = len(time)

    start, end = time[0], time[-1]
    new_time = np.linspace(start, end, length * dense_factor, endpoint=True)
    new_freq = np.interp(new_time, time, freq)
    return np.stack([new_time, new_freq], axis=-1)


def tf_to_pix(
    traj: np.ndarray,
    num_freq_bins: int = NUM_FREQ_BINS,
    width: int = N_FRAMES,
):
    """Convert time-frequency coordinates to pixel coordinates within a single spectrogram segment

    Args:
        traj: time-frequency coordinates of the contour [(time(s), frequency(Hz))]: (num_points, 2)

    Returns:
        pixel coordinates of the contour [(column, row)] in int, left bottom origin: (num_points, 2)
    """
    times = traj[:, 0]
    freqs = traj[:, 1]
    columns = times * FRAME_PER_SECOND + 0.5
    row_top = freqs / FREQ_BIN_RESOLUTION
    rows = num_freq_bins - row_top
    rows = np.round(rows - 0.5).astype(int)
    columns = np.round(columns).astype(int)
    coords = np.unique(np.stack([columns, rows], axis=-1), axis=0) # remove duplicate points
    valid_mask  = (coords[:, 0] >= 0) & (coords[:, 0] < width) & (coords[:, 1] >= 0) & (coords[:, 1] < num_freq_bins)
    return coords[valid_mask]


def polyline_to_polygon(traj: np.ndarray, width: float = 3)-> List[float]:
    """Convert polyline to polygon
    
    Args:
        traj: polyline coordinates [(column, row)]: (num_points, 2) in single spec segment
        width: width of the polyline

    Returns:
        coco segmentation format [x1, y1, x2, y2, ...]
    """
    if len(traj) == 0:
        return False
    if len(traj) == 1:
        # For single point, create a circular polygon
        point = Point(traj[0])
        polygon = point.buffer(width / 2)
    else:
        # Original logic for polyline
        line = LineString(traj)
        polygon = line.buffer(width / 2)

    if polygon.geom_type == "MultiPolygon":
        raise ValueError("The trajectory is too wide, resulting in multiple polygons")
    
    polygon = clip_by_rect(polygon, 0, 0, N_FRAMES, NUM_FREQ_BINS)

    if polygon.is_empty:
            return []
    if polygon.geom_type == "MultiPolygon":
        raise ValueError("The trajectory is too wide, resulting in multiple polygons")

    coco_coords = np.array(polygon.exterior.coords).round(2)
    if len(coco_coords) < 3:
        return []
    return coco_coords.ravel().tolist()  # coco segmentation format [x1, y1, x2, y2, ...]

def polygon_to_box(polygon: List[float]):
    """Convert polygon to bounding box in format xywh
    
    Args:
        polygon: coco segmentation format [x1, y1, w, h]
    """
    x = polygon[::2]
    y = polygon[1::2]
    x1, x2 = min(x), max(x)
    y1, y2 = min(y), max(y)
    return [x1, y1, x2 - x1, y2 - y1]  # [x, y, w, h]


def xyxy2xywh(bbox: list) -> list:
    """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
    evaluation.

    Args:
        bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
            ``xyxy`` order.

    Returns:
        list[float]: The converted bounding boxes, in ``xywh`` order.
    """

    return [
        bbox[0],
        bbox[1],
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
    ]


def get_segment_annotation(trajs: List[np.array], start_frame:int):
    """Get the part of annotations within the segment range
    
    Args:
        trajs: list of contours [(time(s), frequency(Hz))]: [(num_points, 2),...]
        start_frame: start frame of the segment
    Returns:
        segment_trajs: list of contours within the segment range [(time(s), frequency(Hz))]: [(num_points, 2),...]
    """
    # determine the range of segments
    start_time = start_frame * HOP_MS / 1000
    end_time = (start_frame + N_FRAMES) * HOP_MS / 1000

    segment_trajs = []
    for traj in trajs:
        if traj[0, 0] <= end_time and traj[-1, 0] >= start_time:
            mask = (traj[:, 0] >= start_time) & (traj[:, 0] <= end_time)
            segment_traj = traj[mask]
            # get the traj whitin range, relative to the segment
            segment_traj[:, 0] = segment_traj[:, 0] - start_time
            segment_trajs.append(segment_traj)
    return segment_trajs

