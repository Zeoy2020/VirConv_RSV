CUDA_VISIBLE_DEVICES=0 python test.py \
    --cfg_file cfgs/kitti_models/VirConv-L.yaml \
    --ckpt ../output/kitti_models/VirConv-L/default/ckpt/checkpoint_epoch_3.pth \
    --batch_size 16 \