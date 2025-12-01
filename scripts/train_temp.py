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
import pytorch_lightning as pl

from diffusion.resample import create_named_schedule_sampler
from diffusion import dist_util, logger
from diffusion.tacm_script_temp_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from diffusion.tacm_train_temp_util import TrainLoop
from diffusion.dist_util import save_video_grid
from tacm import VideoData
from einops import rearrange, repeat

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    new_model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    # load original model parameters
    original_model_dict = dist_util.load_state_dict(args.model_path, map_location="cpu")
    # get new model parameters dictionary
    new_model_dict = new_model.state_dict()
    # delete keys in original parameters which are not same as new_model
    pretrained_dict = {k: v for k, v in original_model_dict.items() if k in new_model_dict}
    # keep original parameters (spatial layers) fixed
    for v in original_model_dict.values():
        v.requires_grad = False
    new_model_dict.update(pretrained_dict)
    new_model.load_state_dict(new_model_dict)

    #print(new_model)

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)
    
    logger.log("loading dataset...")
    data = VideoData(args)
    data = data.train_dataloader()

    logger.log("training...")
    TrainLoop(
        model=new_model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        save_dir=args.save_dir,
        vqgan_ckpt=args.vqgan_ckpt,
        sequence_length=args.sequence_length,
        audio_emb_model=args.audio_emb_model
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        model_path="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        # batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = VideoData.add_data_specific_args(parser)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--vqgan_ckpt', type=str)
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
