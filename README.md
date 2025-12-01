# TIA2V: Video Generation Conditioned on Triple Modalities of Text-Image-Audio
This is the official implement of our proposed method of TIA2V task. As a progressive development of our previous work TA2V, in this paper, we combine text, image and audio reasonably and effectively through a single diffusion model as composable conditions, to generate more controllable and customized videos, which will be generalized among all kinds of dataset.

<img width="800" alt="model" src="https://github.com/user-attachments/assets/1e7cc394-c7bb-419a-ac19-f19113f057e3">

## Examples
### without SHR module
https://github.com/user-attachments/assets/f6a584d4-a2da-4cad-b7a7-2c91c0cb028c

https://github.com/user-attachments/assets/7b05d22d-43e0-4616-a9c7-81708005ddc6

https://github.com/user-attachments/assets/c4938c6d-0829-4274-aaae-8f83e6033243

https://github.com/user-attachments/assets/5a06fbeb-6fef-4001-928a-d4a155dfd2ee

### with SHR module
https://github.com/user-attachments/assets/4c47b7e4-a286-467d-9651-32bd9ef5338e

https://github.com/user-attachments/assets/dbc2f84a-08b9-4dce-92ac-6574c9a6efaf

https://github.com/user-attachments/assets/1ac1de47-3c2f-4d65-81aa-ce8f9cb5d07b

https://github.com/user-attachments/assets/f5808493-cf2d-40d8-ac2e-eceb28264562

## Setup
1. Create the virtual environment
```bash
conda create -n tia python==3.9
conda activate tia
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
pip install pytorch-lightning==1.5.4 einops ftfy h5py imageio regex scikit-image scikit-video tqdm lpips blobfile mpi4py opencv-python-headless kornia termcolor pytorch-ignite visdom piq joblib av==10.0.0 matplotlib ffmpeg==4.2.2 pillow==9.5.0
pip install git+https://github.com/openai/CLIP.git wav2clip transformers
```
2. Create a `saved_ckpts` folder to download pretrained checkpoints.

## Datasets
We create three three-modality datasets named as [URMP-VAT](https://studentmust-my.sharepoint.com/:u:/g/personal/3220000901_student_must_edu_mo/ESzQSC36S_dAnOtOyiuaCZ8BmhJeT1CAHnm2Wyc5Z3lDWA), [Landscape-VAT](https://studentmust-my.sharepoint.com/:u:/g/personal/3220000901_student_must_edu_mo/EeVm6iowmChMuuCkGe0ApSYBFbYxI7EbLW7eJhqBIOi0OQ?e=sOw8eG), and [AudioSet-Drums-VAT].


## Download pre-trained checkpoints
| Dataset | TIA | Diffusion |
| --------------- | --------------- | --------------- |
| URMP-VAT | [URMP-VAT_tia.pt](https://drive.google.com/file/d/1T_SovYzSnyn43KLnNkSb49tJ9Ju3RLkO/view?usp=sharing) | [URMP-VAT_diffusion.pt](https://drive.google.com/file/d/1Hg7A5eigOZ_TzM5hfJqGiau04usi3k8-/view?usp=drive_link)
| Landscape-VAT | [Landscape-VAT_tia.pt](https://drive.google.com/file/d/1yU-_angGLe9a2feLMJq2n20WXaQoagno/view?usp=drive_link) | [Landscape-VAT_diffusion.pt](https://drive.google.com/file/d/1aYoXYxJpcFpP7Eq5kwCusXDxWl-LPMAr/view?usp=drive_link)
| AudioSet-Drums-VAT | [AudioSet-Drums-VAT_tia.pt](https://drive.google.com/file/d/1vMAUcTSyUYqeFgTm9yKu09UhJHl3JzSx/view?usp=drive_link) | [AudioSet-Drums-VAT_diffusion.pt](https://drive.google.com/file/d/1tg7pCLF_TvXRwqyfcIRdUpluJsXAO4lj/view?usp=drive_link)

## Sampling Procedure
### Sample Short Music Performance Videos
- `data_path`: path to dataset, default is `post_URMP`
- `text_emb_model`: model to encode text, choices: `bert`, `clip`
- `audio_emb_model`: model to encode audio, choices: `audioclip`, `wav2clip`
- `text_stft_cond`: load text-audio-video data
- `n_sample`: the number of videos need to be sampled
- `run`: index for each run
- `resolution`: resolution to extract data
- `model_path`: the path of pre-trained checkpoint
- `image_size`: the resolution used in training process
- `in_channels`: the number of channels of the input videos/frames
- `diffusion_steps`: the number of steps to denoise
- `noise_schedule`: choices: `cosine`, `linear`
- `num_channels`: latent channels base
- `num_res_blocks`: the number of resnet blocks in diffusion
```
python scripts/sample_motion_optim.py --resolution 64 --batch_size 1 --diffusion_steps 4000 --noise_schedule cosine \
--num_channels 64 --num_res_blocks 2 --class_cond False --model_path saved_ckpts/your_model.pt \
--num_samples 50 --image_size 64 --learn_sigma True --text_stft_cond --audio_emb_model beats --data_path datasets/post_URMP \
--in_channels 3 --clip_denoised True --run 0
```

## Training Procedure
You can also train the models on customized datasets. Here we provide the command to train content and motion parts individually.
### train content
- `save_dir`: path to save checkpoints
- `diffusion_steps`: the number of steps to denoise
- `noise_schedule`: choices: `cosine`, `linear`
- `num_channels`: latent channels base
- `num_res_blocks`: the number of resnet blocks in diffusion
- `class_cond`: whether using class or not
- `image_size`: resolution of videos/images
- `sequence_length`: the number of frames unsed in training
- `lr`: the learning rate
```
python scripts/train_content.py --num_workers 8 --gpus 1 --batch_size 1 --data_path datasets/post_URMP/ \
--save_dir saved_ckpts/your_directory_path --resolution 64 --sequence_length 16 --text_stft_cond --diffusion_steps 4000 \
--noise_schedule cosine --lr 5e-5 --num_channels 64 --num_res_blocks 2 --class_cond False --log_interval 50 \
--save_interval 10000 --image_size 64 --learn_sigma True --in_channels 3
```
### train motion
- `save_dir`: path to save checkpoints
- `diffusion_steps`: the number of steps to denoise
- `noise_schedule`: choices: `cosine`, `linear`
- `num_channels`: latent channels base
- `num_res_blocks`: the number of resnet blocks in diffusion
- `class_cond`: whether using class or not
- `image_size`: resolution of videos/images
- `sequence_length`: the number of frames unsed in training
- `model_path`: the path of content model
- `audio_emb_model`: model to encode audio, choices: `audioclip`, `wav2clip`
```
python scripts/train_temp.py --num_workers 8 --batch_size 1 --data_path datasets/post_URMP/ \
--model_path saved_ckpts/your_content_model.pt --save_dir saved_ckpts/your_directory_path --resolution 64 \
--sequence_length 16 --text_stft_cond --audio_emb_model beats --diffusion_steps 4000 --noise_schedule cosine \
--num_channels 64 --num_res_blocks 2 --class_cond False --image_size 64 --learn_sigma True --in_channels 3 \
--lr 5e-5 --log_interval 50 --save_interval 5000 --gpus 1
```

## Acknowledgements
Our code is based on [Latent-Diffusion](https://github.com/CompVis/latent-diffusion).
