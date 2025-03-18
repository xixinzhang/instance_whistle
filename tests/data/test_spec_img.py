import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.path import Path
from skimage.draw import polygon2mask

from whistle_prompter.datasets.prepare_spec_img import *


def polygon_anno_to_mask(img_path, annotations_data, output_path=None, mask_color=(1, 0, 0), alpha=0.5):

    images = annotations_data['images']
    annos = annotations_data['annotations']

    image_info = images[0]
    img_file = image_info['file_name']
    img = plt.imread(os.path.join(img_path, img_file ))
    height, width = img.shape[:2]
    
    image_id = image_info['id']
    annotations = [anno for anno in annos if anno['image_id'] == image_id]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    
    # Process each annotation
    for anno in annotations:
        if 'segmentation' in anno:
            polygons = anno['segmentation']
            # Handle both COCO formats: [[x1,y1,x2,y2,...]] and [x1,y1,x2,y2,...]
            if isinstance(polygons, list):
                if isinstance(polygons[0], list):
                    # Format: [[x1,y1,x2,y2,...], [another polygon], ...]
                    for poly in polygons:
                        points = np.array(poly).reshape(-1, 2)
                        # Create mask for this polygon
                        poly_mask = polygon2mask((height, width), points)
                        mask |= poly_mask
                        
                        # Draw polygon boundary
                        patch = Polygon(points, closed=True,
                                      edgecolor=mask_color,
                                      facecolor='none',
                                      linewidth=2)
                        ax.add_patch(patch)
                # else:
                #     # Format: [x1,y1,x2,y2,...]
                #     points = np.array(polygons).reshape(-1, 2)
                #     poly_mask = polygon2mask((height, width), points)
                #     mask |= poly_mask
                    
                #     # Draw polygon boundary
                #     patch = Polygon(points, closed=True,
                #                   edgecolor=mask_color,
                #                   facecolor='none',
                #                   linewidth=2)
                #     ax.add_patch(patch)
        elif 'bbox' in anno:
            # Handle bbox format: [x, y, width, height]
            x, y, w, h = anno['bbox']
            # Convert bbox to mask
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            mask[y1:y2, x1:x2] = 1
            
            # Draw rectangle
            rect = plt.Rectangle((x, y), w, h, 
                               edgecolor=mask_color,
                               facecolor='none',
                               linewidth=2)
            ax.add_patch(rect)
    
    # Create a colored mask for overlay
    colored_mask = np.zeros((height, width, 4), dtype=np.float32)
    colored_mask[mask == 1] = [mask_color[0], mask_color[1], mask_color[2], alpha]
    
    # Overlay the mask on the image
    ax.imshow(colored_mask)
    
    ax.set_title('Image with Mask Overlay')
    ax.axis('off')
    plt.tight_layout()
    
    # Save the result if output_path is provided
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=300)
        print(f"Saved result to {output_path}")
    
    return fig, mask
if __name__ == "__main__":
    filesname = ["data/audio/palmyra092007FS192-070924-205305.wav"]
    segments_dict = audios_to_segments_dict(filesname)
    print(segments_dict['palmyra092007FS192-070924-205305'].keys())
    output_dir = "outputs/test_spec_anno/"
    os.makedirs(output_dir, exist_ok=True)
    save_specs_img(segments_dict, output_dir, cmap=None, line_width=3)
    anno_file = os.path.join(output_dir, "annotations.json") 
    with open(anno_file) as f:
        annotations_data = json.load(f)

    polygon_anno_to_mask(output_dir, annotations_data, output_path=os.path.join(output_dir, "a_mask_overlay.png"))

