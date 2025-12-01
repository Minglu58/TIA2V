#!/bin/bash
CUDA_VISIBLE_DEVICES=0 \
python scripts/train_diffusion_3d_withtext.py --num_workers 8 \
                                --gpus 1 \
                                --batch_size 2 \
                                --data_path datasets/post_landscape/ \
                                --save_dir saved_ckpts/landscape/diffusion_0904\
                                --resolution 128 \
                                --sequence_length 16 \
                                --text_stft_cond \
                                --diffusion_steps 4000 \
                                --noise_schedule cosine \
                                --lr 5e-5 \
                                --num_channels 64 \
                                --num_res_blocks 2 \
                                --class_cond False \
                                --log_interval 50 \
                                --save_interval 10000 \
                                --image_size 128 \
                                --learn_sigma True \
                                --in_channels 3 \
                                --dims 3 \
                                --resume_checkpoint saved_ckpts/landscape/diffusion_0904/model060000.pt
