#!/bin/bash
CUDA_VISIBLE_DEVICES=1 \
python scripts/sample_motion.py --resolution 64 \
                                --batch_size 1 \
                                --diffusion_steps 4000 \
                                --noise_schedule cosine \
                                --num_channels 64 \
                                --num_res_blocks 2 \
                                --class_cond False \
                                --model_path saved_ckpts/urmp/tia_0906/model110000.pt \
                                --num_samples 5 \
                                --image_size 64 \
                                --learn_sigma True \
                                --text_stft_cond \
                                --audio_emb_model beats \
                                --data_path datasets/post_URMP \
                                --load_vid_len 90 \
                                --in_channels 3 \
                                --clip_denoised True
#--predict_xstart True
#--use_ddim --timestep_respacing 4000
