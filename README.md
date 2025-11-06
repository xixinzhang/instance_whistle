# Whistle instance segmentation with learned prompt

## Installation
```shell
cd instance_whistle
pip install -e .
pip install mmengine
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html  # torch version is 2.4.0
cd ../mmdetection
pip install -e .
```
## Prepare Dataset
1. dataset structure
```shell
data
├── anno
├── audio
├── meta.yaml
```
2. prepare dataset meta.yaml looks like this:
```yaml
test:
- Qx-Tt-SCI0608-N1-060814-121518
- QX-Dc-CC0604-TAT25-060413-220000
- Qx-Tt-SCI0608-N1-060814-123433
train:
- palmyra092007FS192-070928-040000
- palmyra102006-061020-204454_4
```
3. save image dataset with coco format annotations
```bash
python whistle_prompter/datasets/prepare_spec_img.py \
    --meta path/to/meta.yaml \
    --anno_dir anno \
    --audio audio \
    --output_dir path/to/output
```

## Training


## Inference
