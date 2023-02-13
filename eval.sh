CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
  --nproc_per_node=1 train_cad_ddp.py \
  --data_root /home/chli/chLi/FloorPlanCAD \
  --pretrained_model /home/chli/chLi/HRNet/hrnetv2_w48_imagenet_pretrained.pth \
  --test_only

# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch \
  # --nproc_per_node=1 train_cad_ddp.py \
  # --data_root /home/chli/chLi/FloorPlanCAD \
  # --pretrained_model /home/chli/chLi/HRNet/hrnetv2_w48_imagenet_pretrained.pth \
  # --val_only
