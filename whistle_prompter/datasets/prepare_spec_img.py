import json
import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from tqdm import tqdm

from whistle_prompter import utils


def audios_to_segments_dict(filenames:List[str])-> dict[str, dict[int, np.ndarray]]:
    """Prepare spectrogram segments from audio files

    Args:
        filenames: audio file paths
    Return:
        segments_dict: {stem: {start_frame: segment}}
    """

    if not isinstance(filenames, List):
        filenames = [filenames]
    if not isinstance(filenames[0], str):
        if isinstance(filenames[0], Path):
            filenames = [str(f) for f in filenames]
        else:
            raise TypeError("filenames must be a list of strings")
    
    segments_dict = dict()
    for f in tqdm(filenames):
        waveform = utils.load_audio(f)
        spec = utils.spectrogram(waveform)
        stem = f.split("/")[-1].split(".")[0]
        normalized_spec = utils.normalize_spec_img(spec)
        segments_dict.update({stem: utils.cut_sepc(normalized_spec)}) # {stem: {start_frame: segment}}
    return segments_dict

def save_specs_img(segments_dict:dict[str, dict[int, np.ndarray]], save_dir:str, line_width, cmap:Optional[str]):
    """Save spectrogram images from audio files

    Args:
        segments_dict: {stem: {start_frame: segment}}
        save_dir: directory to save the images
    """
    img_cnt = 0
    whistle_cnt = 0
    annotations = []
    images = []

    for stem, segments in segments_dict.items():
        print(f"Saving {stem} to {save_dir}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        bin_file = Path(f'data/raw_anno/{stem}.bin')
        annos = utils.load_annotation(bin_file)
        for start_frame, segment in segments.items():
            spec_img_name = f"{stem}_{start_frame}.png"
            # images
            images.append(dict(
                id=img_cnt,
                width = segment.shape[1],
                height = segment.shape[0],
                file_name = spec_img_name,
                audio_filename = stem,
                start_frame = start_frame
            ))
            img_cnt+=1
            # annotations
            trajs = utils.get_segment_annotation(annos, start_frame)
            for traj in trajs:
                traj_pix = utils.tf_to_pix(traj)
                traj_plg = utils.polyline_to_polygon(traj_pix, width=line_width)
                if not traj_plg: # skip polygon with less than 3 points
                    continue
                bbox = utils.polygon_to_box(traj_plg)
                annotations.append(dict(
                    id=whistle_cnt,
                    image_id=img_cnt,
                    category_id=0,
                    segmentation = [traj_plg],
                    area = bbox[2]*bbox[3],
                    bbox = bbox,
                    iscrowd = 0
                ))
                whistle_cnt+=1

            # Save the image
            if cmap is not None:
                segment = utils.apply_colormap(segment, cmap)
            else:
                segment = np.stack([segment]*3, axis=-1) # F, N_FRAMES, 3)
            segment = (segment*255).astype(np.uint8)
            segment = cv2.cvtColor(segment, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{save_dir}/{spec_img_name}", segment)
            print(f"Saved {save_dir}/{spec_img_name}")
        
    # Save the annotations
    category = dict(id=0, name="whistle")
    coco_json = dict(
        images=images,
        annotations=annotations,
        categories=[category]
    )
    with open(f"{save_dir}/annotations.json", "w") as f:
        json.dump(coco_json, f)
        print(f"Saved {save_dir}/annotations.json")



def split_specs_dataset():
    pass



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmap", type=str, default=None)
    parser.add_argument("--line_width", type=float, default=3)
    args = parser.parse_args()

    with open("data/meta.json") as f:
        meta = json.load(f)
    filenames = []
    for _, stems in meta["data"].items():
        filenames.extend([f"data/audio/{stem}.wav" for stem in stems])
    segments_dict = audios_to_segments_dict(filenames)
    print(segments_dict.keys())
    save_specs_img(segments_dict, f"data/spec_img/", cmap=args.cmap, line_width=args.line_width)

    