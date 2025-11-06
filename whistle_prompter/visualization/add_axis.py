from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from pycocotools.coco import COCO
from pathlib import Path

plt.rcParams['font.family'] = 'Times New Roman'

parser = argparse.ArgumentParser()
parser.add_argument('dir', type=Path, help='Path to the image directory')
parser.add_argument('out_dir', type=Path, help='Path to the output directory')
args = parser.parse_args()

# Create output directory if it doesn't exist
args.out_dir.mkdir(parents=True, exist_ok=True)

for file in args.dir.iterdir():
    if file.is_file():
        if not file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        img = plt.imread(file)
        
        plt.imshow(img, aspect='auto')
        x_ticks = np.linspace(0, img.shape[1], num=6, endpoint=True, dtype=int)
        x_labels = np.around(np.linspace(0, 3, 6), decimals=2)
        y_ticks = np.linspace(0, img.shape[0], num=7, endpoint=True, dtype=int)
        y_labels = np.around(np.linspace(0, 96, 7)[::-1], decimals=2)

        plt.figure(figsize=(6, 3))
        plt.imshow(img, aspect='auto')
        plt.xlabel('Time (s)', fontsize=14)
        plt.ylabel('Frequency (kHz)', fontsize=14)
        plt.xticks(x_ticks, x_labels, fontsize=12)
        plt.yticks(y_ticks, y_labels, fontsize=12)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(f'{args.out_dir}/{file.stem}.png', dpi=300)
        plt.close()
