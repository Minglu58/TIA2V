#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python scripts/sample_content.py \
                                  --batch_size 1 \
                                  --diffusion_steps 4000 \
                                  --noise_schedule cosine \
                                  --num_channels 64 \
                                  --num_res_blocks 2 \
                                  --class_cond False \
                                  --model_path saved_ckpts/landscape/ti_0903/model1070000.pt \
                                  --num_samples 5 \
                                  --image_size 64 \
                                  --learn_sigma True \
                                  --data_path datasets/post_landscape \
                                  --text_stft_cond  \
                                  --resolution 64 \
                                  --sequence_length 16 \
#--use_ddim --timestep_respacing 4000
