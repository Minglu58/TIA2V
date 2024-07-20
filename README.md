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
We create two three-modality datasets named as [URMP-VAT](https://drive.google.com/file/d/1u8dA_TwivVj83DEr74Yw_bLPcOevEHb2/view?usp=sharing).

## Download pre-trained checkpoints
coming

## Sampling Procedure
### Sample Short Music Performance Videos
- `data_path`: path to dataset, you can change it to `post_landscape` for Landscape-VAT dataset
- `text_emb_model`: model to encode text, choices: `bert`, `clip`
- `audio_emb_model`: model to encode audio, choices: `audioclip`, `wav2clip`
- `text_stft_cond`: load text-audio-video data
- `n_sample`: the number of videos need to be sampled
- `run`: index for each run
- `resolution`: resolution used in training video VQGAN procedure
- `model_path`: the path of pre-trained checkpoint
- `image_size`: the resolution used in training process
- `in_channels`: the number of channels of the input image
```
python scripts/sample_motion_optim.py --resolution 64 --batch_size 1 --diffusion_steps 4000 --noise_schedule cosine --num_channels 64 --num_res_blocks 2 --class_cond False --model_path saved_ckpts/diffusion_disc/model110000.pt --num_samples 50 --image_size 64 --learn_sigma True --text_stft_cond --audio_emb_model beats --data_path datasets/post_URMP --in_channels 3 --clip_denoised True --run 0
```
