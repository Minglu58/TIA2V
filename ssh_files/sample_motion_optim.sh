#!/bin/bash
CUDA_VISIBLE_DEVICES=3 \
python scripts/sample_motion_optim.py --resolution 64 \
                                      --batch_size 1 \
                                      --diffusion_steps 4000 \
                                      --noise_schedule cosine \
                                      --num_channels 64 \
                                      --num_res_blocks 2 \
                                      --class_cond False \
                                      --model_path saved_ckpts/landscape/tia_0905/model170000.pt \
                                      --num_samples 50 \
                                      --image_size 64 \
                                      --learn_sigma True \
                                      --text_stft_cond \
                                      --audio_emb_model beats \
                                      --data_path datasets/post_landscape \
                                      --dataset landscape \
                                      --load_vid_len 30 \
                                      --in_channels 3 \
                                      --clip_denoised True \
                                      --run 1 \
                                      --diffusion_ckpt saved_ckpts/landscape/diffusion_0904/model150000.pt
#--predict_xstart True
#--use_ddim --timestep_respacing 4000
