#!/bin/bash

python scripts/train_cm_vqgan.py --embedding_dim 4 --n_codes 16384 --n_hiddens 32 --downsample 1 8 8 \
--no_random_restart --gpus 0 --batch_size 8 --num_workers 8 --accumulate_grad_batches 6 \
--progress_bar_refresh_rate 50 --max_steps 200000 --gradient_clip_val 1.0 --lr 6.0e-5 \
--data_path datasets/post_URMP/ --default_root_dir saved_ckpts/vqgan64_0219 --resolution 64 --sequence_length 16 \
--discriminator_iter_start 10000 --norm_type batch --perceptual_weight 4 --image_gan_weight 1 \
--video_gan_weight 1  --gan_feat_weight 4 \
#--resume_from_checkpoint saved_ckpts/epoch=5_val_recon_loss=0.20.ckpt