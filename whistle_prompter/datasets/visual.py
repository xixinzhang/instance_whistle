from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from pycocotools.coco import COCO

plt.rcParams['font.family'] = 'Times New Roman'

parser = argparse.ArgumentParser()
parser.add_argument('file', type=str, help='Path to the image file')
parser.add_argument('--gt', type=str, help='Path to the ground truth mask file')
args = parser.parse_args()

gt_path = args.gt if args.gt else None

# Create output directory if it doesn't exist
os.makedirs('outputs/vis', exist_ok=True)


img_path = f'data/cross_coco/val/data/{args.file}'
# Load the original image
img = plt.imread(img_path)

# Set up ticks and labels
x_ticks = np.linspace(0, img.shape[1], num=10, endpoint=True, dtype=int)
x_labels = np.around(np.linspace(0, 3, 10), decimals=2)
y_ticks = np.linspace(0, img.shape[0], num=10, endpoint=True, dtype=int)
y_labels = np.around(np.linspace(0, 96, 10)[::-1], decimals=2)

# Image 1: Raw image with all axes
plt.figure(figsize=(10, 6))
plt.imshow(img, aspect='auto')
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Frequency (kHz)', fontsize=14)
plt.xticks(x_ticks, x_labels, fontsize=12)
plt.yticks(y_ticks, y_labels, fontsize=12)
plt.tight_layout()
plt.savefig(f'outputs/vis/{os.path.splitext(args.file)[0]}_raw.png', dpi=300)
plt.close()

# Image 2: Original visualization (same as before)
img_path = f'mmdetection/outputs/vis/{args.file}'
img = plt.imread(img_path)

plt.figure(figsize=(10, 6))
plt.imshow(img, aspect='auto')
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Frequency (kHz)', fontsize=14)
plt.xticks(x_ticks, x_labels, fontsize=12)
plt.yticks(y_ticks, y_labels, fontsize=12)
plt.tight_layout()
plt.savefig(f'outputs/vis/{os.path.splitext(args.file)[0]}_dt.png', dpi=300)
plt.close()

# Image 3: Ground truth mask overlayed
fig, ax = plt.subplots(figsize=(10, 6))
img_path = f'data/cross_coco/val/data/{args.file}'
img = plt.imread(img_path)
plt.imshow(img, aspect='auto')
val_coco = COCO('data/cross_coco/val/labels.json')

for img_id, img in val_coco.imgs.items():
    if img['file_name'] == args.file:
        img_info = img
print(img_info)
annIds = val_coco.getAnnIds(imgIds=img_info['id'],  iscrowd=None)
anns = val_coco.loadAnns(annIds)
polygons = []
color = []
for ann in anns:
    c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
    for seg in ann['segmentation']:
        poly = np.array(seg).reshape((int(len(seg)/2), 2))
        polygons.append(Polygon(poly))
        color.append(c)
    p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
    ax.add_collection(p)
    p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=0.5)
    ax.add_collection(p)

plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Frequency (kHz)', fontsize=14)
plt.xticks(x_ticks, x_labels, fontsize=12)
plt.yticks(y_ticks, y_labels, fontsize=12)
plt.tight_layout()
plt.savefig(f'outputs/vis/{os.path.splitext(args.file)[0]}_gt.png', dpi=300)
plt.close()
