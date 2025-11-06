from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from pycocotools.coco import COCO
from pathlib import Path

plt.rcParams['font.family'] = 'Times New Roman'


def high_contrast_colors(n, seed=None):
    import colorsys
    if seed is not None:
        np.random.seed(seed)
    
    colors = []
    for _ in range(n):
        h = np.random.rand()                    # random hue (0–1)
        s = 0.7 + 0.3 * np.random.rand()        # high saturation (avoid gray)
        v = 0.7 + 0.3 * np.random.rand()        # bright (avoid dark)
        rgb = np.array(colorsys.hsv_to_rgb(h, s, v))
        colors.append((rgb * 255).astype(np.uint8))
    return colors


parser = argparse.ArgumentParser()
parser.add_argument('file', type=Path, help='Path to the image file')
parser.add_argument('--out_dir', type=Path, default=Path('outputs/vis'), help='Path to the output directory')
args = parser.parse_args()

# Create output directory if it doesn't exist
args.out_dir.mkdir(parents=True, exist_ok=True)


fig, ax = plt.subplots(figsize=(6, 3))
img = plt.imread(args.file)
cmap = plt.get_cmap('binary')
img = (cmap(img[..., 0])[..., :3] * 255).astype(np.uint8)

# save img
plt.imsave(f'{args.out_dir}/{args.file.stem}_processed.png', img)

plt.imshow(img, aspect='auto')
x_ticks = np.linspace(0, img.shape[1], num=6, endpoint=True, dtype=int)
x_labels = np.around(np.linspace(0, 3, 6), decimals=2)
y_ticks = np.linspace(0, img.shape[0], num=7, endpoint=True, dtype=int)
y_labels = np.around(np.linspace(0, 96, 7)[::-1], decimals=2)

ann_file = args.file.parents[1] / 'labels.json'
val_coco = COCO(ann_file.as_posix())

for img_id, img in val_coco.imgs.items():
    if img['file_name'] == args.file.name:
        img_info = img
print(img_info)

# Configure axes before saving each visualization variant so layout stays identical.
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Frequency (kHz)', fontsize=14)
plt.xticks(x_ticks, x_labels, fontsize=12)
plt.yticks(y_ticks, y_labels, fontsize=12)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(args.out_dir / f'{args.file.stem}_raw.png', dpi=300)


annIds = val_coco.getAnnIds(imgIds=img_info['id'],  iscrowd=None)
anns = val_coco.loadAnns(annIds)
polygons = []
# color = []
c = high_contrast_colors(len(anns))
color = [c_ / 255.0 for c_ in c]
for ann in anns:
    # c = (np.random.random((1, 3))*0.6+0.4).tolist()[0]
    for seg in ann['segmentation']:
        poly = np.array(seg).reshape((int(len(seg)/2), 2))
        polygons.append(Polygon(poly))
        # color.append(c)
    p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
    ax.add_collection(p)
    p = PatchCollection(polygons, facecolor='none', edgecolors=color, linewidths=0.5)
    ax.add_collection(p)

plt.tight_layout()
plt.savefig(f'{args.out_dir}/{args.file.stem}_gt.png', dpi=300)
plt.close()
