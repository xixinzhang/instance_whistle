mkdir -p outputs/eval_mask2former_swin
for t in $(seq 0.1 0.1 0.9); do
  echo "Running thre_norm=$t at $(date)"
  python tools/test.py ../configs/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_whistle_noresize_cls11_median.py \
    work_dirs/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_whistle_noresize_cls11_median/iter_368750.pth \
    --data-dir ../data \
    --filter-dt $t \
    --duration 5543.423265306123 > outputs/eval_mask2former_swin/run_$t.log 2>&1
  echo "Finished thre_norm=$t at $(date)"
done

mkdir -p outputs/eval_mask2former_r50
for t in $(seq 0.1 0.1 0.9); do
  echo "Running thre_norm=$t at $(date)"
  python tools/test.py ../configs/mask2former_r50_8xb2-lsj-50e_whistle_cross_median.py \
    work_dirs/mask2former_r50_8xb2-lsj-50e_whistle_cross_median/iter_368750.pth \
    --data-dir ../data \
    --filter-dt $t \
    --duration 5543.423265306123 > outputs/eval_mask2former_r50/run_$t.log 2>&1
  echo "Finished thre_norm=$t at $(date)"
done

mkdir -p outputs/eval_solov2
for t in $(seq 0.1 0.1 0.9); do
  echo "Running thre_norm=$t at $(date)"
  python tools/test.py ../configs/solov2_r50_fpn_ms-3x_whistle_median.py \
    work_dirs/solov2_r50_fpn_ms-3x_coco_median/epoch_36.pth \
    --data-dir ../data \
    --filter-dt $t \
    --duration 5543.423265306123 > outputs/eval_solov2/run_$t.log 2>&1
  echo "Finished thre_norm=$t at $(date)"
done

mkdir -p outputs/eval_maskrcnn
for t in $(seq 0.1 0.1 0.9); do
  echo "Running thre_norm=$t at $(date)"
  python tools/test.py ../configs/mask-rcnn_r50-caffe_fpn_ms-3x_whistle_median.py \
    work_dirs/mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco_median/epoch_36.pth \
    --data-dir ../data \
    --filter-dt $t \
    --duration 5543.423265306123 > outputs/eval_maskrcnn/run_$t.log 2>&1
  echo "Finished thre_norm=$t at $(date)"
done

mkdir -p outputs/eval_cascade
for t in $(seq 0.1 0.1 0.9); do
  echo "Running thre_norm=$t at $(date)"
  python tools/test.py ../configs/cascade-mask-rcnn_r50_fpn_ms-3x_whistle.py \
    work_dirs/cascade-mask-rcnn_r50_fpn_ms-3x_whistle/epoch_12.pth \
    --data-dir ../data \
    --filter-dt $t \
    --duration 5543.423265306123 > outputs/eval_cascade/run_$t.log 2>&1
  echo "Finished thre_norm=$t at $(date)"
done

mkdir -p outputs/eval_htc
for t in $(seq 0.1 0.1 0.9); do
  echo "Running thre_norm=$t at $(date)"
  python tools/test.py ../configs/htc_r50_fpn_20e_whistle.py \
    work_dirs/htc_r50_fpn_20e_whistle/epoch_20.pth \
    --data-dir ../data \
    --filter-dt $t \
    --duration 5543.423265306123 > outputs/eval_htc/run_$t.log 2>&1
  echo "Finished thre_norm=$t at $(date)"
done

echo "✅ All runs completed!"