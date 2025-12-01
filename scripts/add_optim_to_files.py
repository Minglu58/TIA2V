"""
Generate a large batch of video samples from a model.
"""
import sys
sys.path.append('/data/workspace/TACM')
import argparse
import os
import time
import skvideo.io
import numpy as np
import torch as th
import torch.distributed as dist

from diffusion import dist_util, logger
from diffusion.tacm_script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from diffusion.dist_util import save_video_grid
from tacm import VideoData
import torch.nn.functional as Func
from tacm.download import load_vqgan
from einops import rearrange, repeat

import transformers.image_transforms
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer, CLIPTokenizer, CLIPTextModel

original_video_path = 'results/0_tacm_landscape/fake1_6fps'
output_video_path = 'results/0_tacm_landscape/fake2_6fps'

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()


    tokenizer = CLIPTokenizer.from_pretrained('tacm/modules/cache/clip-vit-large-patch14')
    transformer = CLIPTextModel.from_pretrained('tacm/modules/cache/clip-vit-large-patch14')
    transformer = transformer.eval()
    for param in transformer.parameters():
        param.requires_grad = False


    for file_name in os.listdir(original_video_path):
        file_path = os.path.join(original_video_path, file_name)
        video_data = skvideo.io.vread(file_path)
        video_tensor = th.from_numpy(video_data).float()
        video_tensor = video_tensor.permute(3,0,1,2) / 255. - 0.5
        print(video_tensor.min(), video_tensor.max())
        video_tensor = Func.interpolate(video_tensor, size=(128, 128), mode='bilinear', align_corners=False)

        txt_path = file_path.replace("fake1_6fps", "txt").replace(".mp4", ".txt").replace("video", "groundtruth")
        text = [line.rstrip() for line in open(txt_path)]
        batch_encoding = tokenizer(text, truncation=True, max_length=77, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to('cpu')
        outputs = transformer(input_ids=tokens)[0]

        c = outputs.to(dist_util.dev()) #torch.Size([1, 77, 768])

        init_image = video_tensor.to(dist_util.dev()).unsqueeze(0)

        logger.log("sampling...")
        t1 = time.time()
        model_kwargs = {}

        sample_fn = (diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop)
        sample = sample_fn(
            model,
            (args.batch_size, args.in_channels, 16, args.image_size, args.image_size),
            c,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            progress=True,
            skip_timesteps=2500,
            init_image=init_image,
        )
        print(sample.min(), sample.max())
        # sample_recon = sample.add(1).div(2).clamp(0, 1)
        sample_recon = sample.clamp(-0.5, 0.5)+0.5
        #sample_recon = th.clamp(sample, 0, 1.0)        
        
        logger.log("save to mp4 format...")
        os.makedirs(output_video_path, exist_ok=True)
        save_video_grid(sample_recon, os.path.join(output_video_path, file_name), 1)

    #dist.barrier()
    logger.log("sampling complete")
    t2 = time.time()
    sampling_time = t2 - t1
    logger.log(f"sampling time: {sampling_time:.2f} seconds.")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10,
        #batch_size=8,
        use_ddim=False,
        model_path="",
        vqgan_ckpt="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser = VideoData.add_data_specific_args(parser)
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
