CUDA_VISIBLE_DEVICES=0 python test.py \
    --cfg_file cfgs/models/kitti/VirConv-L.yaml \
    --ckpt ../output/models/kitti/VirConv-L/default/ckpt/checkpoint_epoch_1.pth \
    --batch_size 16 
