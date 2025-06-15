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
1. Prepare dataset
```bash
python whistle_prompter/datasets/prepare_spec_img.py
```