CUDA_VISIBLE_DEVICES=1 python test.py \
    --cfg_file cfgs/models/kitti/VirConv-L-RSV-DBV.yaml \
    --batch_size 1 \
    --infer_time \
    --ckpt ../output/models/kitti/VirConv-L-RSV-DBV/default/ckpt/checkpoint_epoch_80.pth \
    2>&1 | grep -v "your gpu arch"
    
