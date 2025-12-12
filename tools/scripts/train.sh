CUDA_VISIBLE_DEVICES=1 python train.py \
    --cfg_file cfgs/models/kitti/VirConv-L.yaml \
    --batch_size 8 \
    --epochs 60 \
    2>&1 | grep -v 'your gpu arch'