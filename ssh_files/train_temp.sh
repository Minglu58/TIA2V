#!/bin/bash
CUDA_VISIBLE_DEVICES=2 \
python scripts/train_temp.py --num_workers 8 \
                            --batch_size 1 \
                            --data_path datasets/post_landscape/ \
                            --load_vid_len 30 \
                            --model_path saved_ckpts/landscape/ti_0903/model1090000.pt \
                            --save_dir saved_ckpts/landscape/tia_0908 \
                            --resolution 64 \
                            --sequence_length 16 \
                            --text_stft_cond \
                            --audio_emb_model beats \
                            --diffusion_steps 4000 \
                            --noise_schedule cosine \
                            --num_channels 64 \
                            --num_res_blocks 2 \
                            --class_cond False \
                            --image_size 64 \
                            --learn_sigma True \
                            --in_channels 3 \
                            --lr 5e-5 \
                            --log_interval 50 \
                            --save_interval 5000 \
                            --gpus 1

#--resume_checkpoint saved_ckpts/text_image_audio_0614/model040000.pt