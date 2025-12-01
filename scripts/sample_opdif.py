"""
Generate a large batch of video samples from a model.
"""
import sys
sys.path.append('/data/workspace/TACM')
import argparse
import os
import time
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
from tacm.download import load_vqgan
from einops import rearrange, repeat

import transformers.image_transforms
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

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
    
    # logger.log("loading vqgan model...")
    # first_stage_model = load_vqgan(args.vqgan_ckpt)
    # for p in first_stage_model.parameters():
    #     p.requires_grad = False
    # first_stage_model.codebook._need_init = False
    # first_stage_model.eval()
    # first_stage_model.train = disabled_train
    #first_stage_vocab_size = first_stage_model.codebook.n_codes
    #first_stage_model.to(dist_util.dev())
    
    logger.log("loading dataset...")
    data = VideoData(args)
    data = data.test_dataloader()

    for i in range(args.num_samples):
        batch = data.dataset.__getitem__(i) #sample_id
        c = batch['text'].to(dist_util.dev()) #torch.Size([1, 77, 768])

        init_image = batch['video'].to(dist_util.dev()).unsqueeze(0)

        logger.log("sampling...")
        t1 = time.time()
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            logger.log(classes)
            model_kwargs["y"] = classes
        sample_fn = (diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop)
        sample = sample_fn(
            model,
            (args.batch_size, args.in_channels, 16, args.image_size, args.image_size),
            c,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            progress=True,
            skip_timesteps=200,
            init_image=init_image,
        )
        print(sample.min(), sample.max())
        # sample_recon = sample.add(1).div(2).clamp(0, 1)
        sample_recon = sample.clamp(-0.5, 0.5)+0.5
        #sample_recon = th.clamp(sample, 0, 1.0)        
        
        logger.log("save to mp4 format...")
        os.makedirs(os.path.join("./results", "optim_diffusion"), exist_ok=True)
        save_video_grid(sample_recon, os.path.join("./results", "optim_diffusion", f"video_%d.mp4"%(i)), 1)

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
