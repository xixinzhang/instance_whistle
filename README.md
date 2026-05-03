# Instance Whistle

Instance segmentation of dolphin and whale whistles from spectrograms using Mask2Former with a Swin Transformer backbone, built on [MMDetection](https://github.com/open-mmlab/mmdetection).

## Project Structure

```
instance_whistle/
├── configs/                  # Model configuration files
├── data/                     # Dataset: audio, annotations, COCO splits
├── mmdetection/              # MMDetection framework (modified)
├── scripts/                  # Evaluation shell scripts
├── instance_whistle/         # Core Python package
│   ├── datasets/             #   Data preparation pipeline
│   ├── utils/                #   Audio, annotation, and I/O utilities
│   └── visualization/        #   Plotting and result visualization
├── outputs/                  # Evaluation and visualization outputs
└── tests/                    # Unit tests
```

## Installation

```bash
# 1. Clone the repository
git clone git@github.com:xixinzhang/instance_whistle.git
cd instance_whistle

# 2. Install the instance_whistle package
pip install -e .

# 3. Install MMEngine and MMCV (match your CUDA/torch version)
pip install mmengine
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html

# 4. Install MMDetection
cd mmdetection
pip install -e .
cd ..
```

**Requirements**: Python >= 3.12, PyTorch 2.4.0, CUDA 12.1

## Model Weights

### Pretrained Backbone Weights

The model is initialized from COCO-pretrained weights. Download to `mmdetection/checkpoints/`:

```bash
mkdir -p mmdetection/checkpoints
wget -P mmdetection/checkpoints/ \
    https://download.openmmlab.com/mmdetection/v3.0/mask2former/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco_20220508_091649-01b0f990.pth
```

### Trained Whistle Detection Weights

Download the trained checkpoint from [Google Drive](https://drive.google.com/file/d/1d5V-FPPXXbtRm-H6OmQ81jU4yuHMZxh5/view?usp=sharing) and place it under `mmdetection/work_dirs/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_whistle_noresize_cls11_median/`:

| Config | Checkpoint |
|--------|------------|
| [mask2former_swin-t_whistle](configs/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_whistle_noresize_cls11_median.py) | [iter_368750.pth](https://drive.google.com/file/d/1d5V-FPPXXbtRm-H6OmQ81jU4yuHMZxh5/view?usp=sharing) |

## Data Preparation

### 1. Organize raw data

```
data/
├── audio/          # Raw .wav files (192 kHz)
├── anno_refined/   # Binary annotation files (.bin, silbido format)
└── meta.yaml       # Train/test split definition
```

`meta.yaml` defines which recordings go into train and test splits:

```yaml
train:
- palmyra092007FS192-070928-040000
- palmyra102006-061020-204454_4
test:
- Qx-Tt-SCI0608-N1-060814-121518
- QX-Dc-CC0604-TAT25-060413-220000
```

### 2. Generate spectrogram images with COCO annotations

This converts raw audio into 3-second spectrogram segments (769 x 1500 pixels) with COCO-format instance segmentation annotations:

```bash
python instance_whistle/datasets/prepare_spec_img.py \
    --meta data/meta.yaml \
    --anno_dir anno_refined \
    --audio audio \
    --output_dir coco_refined
```

**Output structure:**

```
data/coco_refined/
├── train/
│   ├── images/       # Spectrogram PNG images
│   └── labels.json   # COCO-format annotations
└── test/
    ├── images/
    └── labels.json
```

**Options:**
- `--cmap`: Colormap for spectrogram images (default: None, grayscale)
- `--line_width`: Annotation line width in pixels (default: 3)
- `--overlap`: Overlap ratio between adjacent segments (default: 0)

## Training

Training uses the MMDetection framework.

```bash
cd mmdetection

python tools/train.py \
    ../configs/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_whistle_noresize_cls11_median.py
```

Resume from a checkpoint:

```bash
python tools/train.py \
    ../configs/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_whistle_noresize_cls11_median.py \
    --resume work_dirs/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_whistle_noresize_cls11_median/latest.pth
```

Training logs and checkpoints are saved under `mmdetection/work_dirs/`.

## Inference and Evaluation

```bash
cd mmdetection

python tools/test.py \
    ../configs/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_whistle_noresize_cls11_median.py \
    work_dirs/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_whistle_noresize_cls11_median/iter_368750.pth \
    --data-dir ../data \
    --filter-dt 0.5
```

**Arguments:**
- `--data-dir`: Path to the dataset root directory
- `--filter-dt`: Detection confidence threshold (default: 0.8)
- `--split`: Dataset split to evaluate — `train`, `val`, or `test` (default: `test`)
- `--save`: Save predicted whistle annotations
- `--model-name`: Model identifier for saved results (default: `mask2former_swin`)

## Visualization

### Ground truth annotations

```bash
python instance_whistle/visualization/ground_truth.py <image_file> \
    --out_dir outputs/vis
```

### Model predictions

```bash
python instance_whistle/visualization/pred_traj.py <image_file> \
    --model mask2former_swin \
    --ann_dir mmdetection/outputs/qualitative_bins
```

## Citation

<!-- TODO: Add citation -->
