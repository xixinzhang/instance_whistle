import glob
import json
import os
import os.path as osp
import random
import shutil
from pathlib import Path
from typing import List, Optional
import pycocotools.mask as maskUtils

import cv2
import numpy as np
from tqdm import tqdm

from whistle_prompter import utils


def audios_to_segments_dict(filenames:List[str], overlap:float = 0)-> dict[str, dict[int, np.ndarray]]:
    """Prepare spectrogram segments from audio files

    Args:
        filenames: audio file paths
        overlap: overlap ratio between segments
    Return:
        segments_dict: {stem: {start_frame: segment}}
        segment: (F, N_FRAMES)
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
        dirname = f.split("/")[-2]
        normalized_spec = utils.normalize_spec_img(spec)
        segments_dict.update({f'{dirname}/{stem}': utils.cut_sepc(normalized_spec, overlap=overlap)}) # {stem: {start_frame: segment}}
    return segments_dict

def save_specs_img(segments_dict:dict[str, dict[int, np.ndarray]], save_dir:str, line_width, cmap:Optional[str], filter_empty_gt:bool = True):
    """Save spectrogram segment images from audio files to the directory with annotations in COCO format

    Args:
        segments_dict: {stem: {start_frame: segment}}
        save_dir: directory to save the images
    """
    img_cnt = 0
    whistle_cnt = 0
    annotations = []
    images = []

    # clear old data
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(osp.join(save_dir, 'data'), exist_ok=True)

    for name, segments in segments_dict.items():
        dirname, stem = name.split("/")
        print(f"Saving {stem} to {save_dir}")

        bin_file = Path(f'data/cross/anno/{stem}.bin')
        annos = utils.load_annotation(bin_file)
        for start_frame, segment in segments.items():
            # annotations
            trajs = utils.get_segment_annotation(annos, start_frame)
            if filter_empty_gt and not trajs:
                print(f"Skip {stem}_{start_frame} due to no annotation")
                continue
            for traj in trajs:
                traj_pix = utils.tf_to_pix(traj)
                traj_plg = utils.polyline_to_polygon(traj_pix, width=line_width)
                if not traj_plg: # skip polygon with less than 3 points
                    continue
                bbox = utils.polygon_to_box(traj_plg)
                annotations.append(dict(
                    id=whistle_cnt,
                    image_id=img_cnt,
                    category_id=1,
                    segmentation = [traj_plg],
                    area=bbox[2] * bbox[3],
                    bbox = bbox,
                    iscrowd = 0
                ))
                whistle_cnt+=1

            # images
            spec_img_name = f"{stem}_{start_frame}.png"
            images.append(dict(
                id=img_cnt,
                width = segment.shape[1],
                height = segment.shape[0],
                file_name = spec_img_name,
                audio_filename = stem,
                start_frame = start_frame
            ))
            img_cnt+=1
            
            # Save the image
            if cmap is not None:
                segment = utils.apply_colormap(segment, cmap)
            else:
                segment = np.stack([segment]*3, axis=-1) # F, N_FRAMES, 3)
            segment = (segment*255).astype(np.uint8)
            segment = cv2.cvtColor(segment, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{save_dir}/data/{spec_img_name}", segment)
            print(f"Saved {save_dir}/data/{spec_img_name}")
        
    # Save the annotations
    category = dict(id=1, name="whistle")
    coco_json = dict(
        images=images,
        annotations=annotations,
        categories=[category]
    )
    with open(f"{save_dir}/labels.json", "w") as f:
        json.dump(coco_json, f)
        print(f"Saved {save_dir}/labels.json")



def split_specs_dataset(original_annots_path, original_image_dir, output_dir, train_ratio=0.8, val_ratio=0.2, test_ratio=0, seed=42):
    """Split the images with coco annotations into train, val, test sets and save to the directory"""
    random.seed(seed)
    # Validate ratios
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-9, "Ratios must sum to 1"

    with open(original_annots_path, 'r') as f:
        data = json.load(f)
    
    images = data['images']
    annotations = data['annotations']
    categories = data['categories']

    # Shuffle the images
    random.shuffle(images)

    # Split the images
    total = len(images)
    train_num, val_num = int(round(total * train_ratio)), int(round(total * val_ratio))

    # clear the output directory
    shutil.rmtree(output_dir, ignore_errors=True)

    splits = {
        'train': images[:train_num],
        'val': images[train_num:train_num + val_num],
        'test': images[train_num + val_num:]
    }

    # Create the output directory
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    # process each split
    for split, split_images in splits.items():
        img_dir = osp.join(output_dir, split, 'data')
        os.makedirs(img_dir, exist_ok=True)

        img_ids = [img['id'] for img in split_images]
        split_annots = [annot for annot in annotations if annot['image_id'] in img_ids]

        split_data = {
            'images': split_images,
            'annotations': split_annots,
            'categories': categories
        }

        # Save the annotations
        annot_file = osp.join(output_dir, split, f'labels.json')
        with open(annot_file, 'w') as f:
            json.dump(split_data, f)
        
        # Copy the images
        for img in split_images:
            img_name = img['file_name']
            src = osp.join(original_image_dir, img_name)
            dst = osp.join(img_dir, img_name)
            shutil.copy(src, dst)
            print(f"Copy {img_name} to {dst}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmap", type=str, default=None)
    parser.add_argument("--line_width", type=float, default=3)
    parser.add_argument("--overlap", type=float, default=0)
    parser.add_argument("--raw_spec", type = str, default="data/spec_img")
    parser.add_argument("--output_dir", type=str, default="data/coco")
    args = parser.parse_args()

    with open("data/cross/meta.yaml") as f:
        import yaml
        meta = yaml.safe_load(f)
    test_filenames = []
    for stem in meta["test"]:
        test_filenames.append(f"data/cross/audio/{stem}.wav")

    # filenames = filenames[:2]
    # filenames = 'data/cross/audio/Qx-Tt-SCI0608-N1-060814-123433.wav'
    # filenames = 'data/cross/audio/Qx-Tt-SCI0608-N1-060814-121518.wav'
    # with open("data/meta.json") as f:
    #     meta = json.load(f)
    # filenames = [f'data/audio/{f}.wav' for f in meta['data']["test"]]
    filenames = test_filenames

    segments_dict = audios_to_segments_dict(filenames)
    print(segments_dict.keys())
    if filenames[0] in test_filenames:
        save_specs_img(segments_dict, args.raw_spec, filter_empty_gt=False, cmap=args.cmap, line_width=args.line_width)
    else:
        save_specs_img(segments_dict, args.raw_spec, filter_empty_gt=True, cmap=args.cmap, line_width=args.line_width)
    # random split dataset 
    # split_specs_dataset(f"{args.raw_spec}/labels.json", f'{args.raw_spec}/data', args.output_dir)