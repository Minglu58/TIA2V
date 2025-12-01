#!/bin/bash

python scripts/train_content.py --num_workers 8 \
                                --gpus 1 \
                                --batch_size 1 \
                                --data_path datasets/post_URMP/ \
                                --save_dir saved_ckpts/ldm_content_a_0617 \
                                --resolution 64 \
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
                                --image_size 64 \
                                --learn_sigma True \
                                --in_channels 3 \
                                --resume_checkpoint saved_ckpts/ldm_content_a_0617/model110000.pt
