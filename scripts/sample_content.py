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
    
    processor = CLIPProcessor.from_pretrained("tacm/modules/cache/clip-vit-large-patch14")
    clipmodel = CLIPModel.from_pretrained("tacm/modules/cache/clip-vit-large-patch14").to(dist_util.dev())
    
    logger.log("loading dataset...")
    data = VideoData(args)
    data = data.test_dataloader()

    for i in range(args.num_samples):
        batch = data.dataset.__getitem__(i) #sample_id
        print(batch['raw_text'])
        ct = batch['text'].to(dist_util.dev()) #torch.Size([1, 77, 768])
        image = batch['video'][:,0]+0.5
        image = image.unsqueeze(0)
        image_cat=None
        for j in range(image.shape[0]):
            image_j = transformers.image_transforms.to_pil_image(image[j])
            image_input = processor(images=image_j, return_tensors="pt", padding=True).to(dist_util.dev())
            with th.no_grad():
                image_features = clipmodel.get_image_features(image_input.pixel_values)

                if image_cat is None:
                    image_cat = image_features.unsqueeze(0)
                else:
                    image_cat = th.concat((image_cat, image_features.unsqueeze(0)), dim=0) #torch.Size([1, 1, 768])

        cti = th.concat((ct,image_cat), dim=1)
        # c = image_cat

        init_video = rearrange(batch['video'], "c t h w -> t c h w") #torch.Size([16, 3, 64, 64])
        zeros = th.zeros_like(init_video)
        init_video[1:,:,:,:] = zeros[1:,:,:,:]
    

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
            (args.batch_size*16, args.in_channels, args.image_size, args.image_size),
            cti,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            progress=True,
            skip_timesteps=10,
            init_image=init_video.to(dist_util.dev()),
            # init_image=None,
        )
        #sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        #sample = sample.permute(0, 2, 3, 4, 1)
        #sample = sample.contiguous()
        #print('samples:', sample.shape) #torch.Size([1, 256, 16, 8, 8])

        sample = rearrange(sample, '(b f) c h w -> b c f h w', f=16)
        sample_recon = th.clamp(sample, -0.5, 0.5)

        logger.log("save to mp4 format...")
        save_video_grid(sample_recon+0.5, os.path.join("./results", "content_diffusion", f"video_%d.mp4"%(i)), 1)

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
