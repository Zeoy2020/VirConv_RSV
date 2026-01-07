CUDA_VISIBLE_DEVICES=0 python test.py \
    --cfg_file cfgs/models/kitti/VirConv-L-DBV.yaml \
    --ckpt ../output/models/kitti/VirConv-L-DBV/default/ckpt/checkpoint_epoch_60.pth \
    --batch_size 32 \
    --save_to_file \
    2>&1 | grep -v 'your gpu arch'
