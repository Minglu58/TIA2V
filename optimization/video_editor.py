import os
from pathlib import Path
from optimization.constants import ASSETS_DIR_NAME, RANKED_RESULTS_DIR

from tacm.utils import MetricsAccumulator,save_video

from numpy import random
from optimization.augmentations import ImageAugmentations

from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from torchvision.transforms import functional as TF
from torch.nn.functional import mse_loss
from optimization.losses import range_loss, d_clip_loss, audioclip_loss
import lpips
import numpy as np

import clip
import wav2clip

from diffusion.tacm_script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
)

from diffusion import logger

from tacm import AudioCLIP
#from utils.transforms import ToTensor1D


class VideoEditor:
    def __init__(self, args) -> None:
        self.args = args
        os.makedirs(self.args.output_path, exist_ok=True)

        self.ranked_results_path = Path(os.path.join(self.args.output_path, RANKED_RESULTS_DIR))
        #os.makedirs(self.ranked_results_path, exist_ok=True)

        if self.args.seed is not None:
            torch.manual_seed(self.args.seed)
            np.random.seed(self.args.seed)
            random.seed(self.args.seed)

        self.model_config = model_and_diffusion_defaults()
        self.model_config.update(
            {
                "attention_resolutions": "16, 8",
                "class_cond": False,
                "diffusion_steps": 4000,
                "rescale_timesteps": True,
                "timestep_respacing": self.args.timestep_respacing,
                "image_size": self.args.model_output_size,
                "in_channels": 3,
                "learn_sigma": True,
                "noise_schedule": self.args.noise_schedule,
                "num_channels": 64,
                "num_head_channels": -1,
                "num_res_blocks": 2,
                "resblock_updown": False,
                "use_fp16": False,
                "use_scale_shift_norm": True,
                "dims": 3,
            }
        )

        # Load models
        self.device = torch.device(
            f"cuda:{self.args.gpu_id}" if torch.cuda.is_available() else "cpu"
        )
        print("Using device:", self.device)

        self.model, self.diffusion = create_model_and_diffusion(**self.model_config)
        self.model.load_state_dict(torch.load(self.args.diffusion_ckpt,map_location="cpu",))
        self.model.requires_grad_(False).eval().to(self.device)
        for name, param in self.model.named_parameters():
            if "qkv" in name or "norm" in name or "proj" in name:
                param.requires_grad_()
        if self.model_config["use_fp16"]:
            self.model.convert_to_fp16()

        self.clip_model = (
            clip.load("ViT-B/32", device=self.device, jit=False)[0].eval().requires_grad_(False)
        )
        self.clip_size = self.clip_model.visual.input_resolution
        self.clip_normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        
        self.wav2clip_model = wav2clip.get_model()
        self.wav2clip_model = self.wav2clip_model.to(self.device)
        for p in self.wav2clip_model.parameters():
            p.requires_grad = False
            
        self.lpips_model = lpips.LPIPS(net="vgg").to(self.device)

        self.image_augmentations = ImageAugmentations(self.clip_size, self.args.aug_num)
        self.metrics_accumulator = MetricsAccumulator()
        
        self.audioclip = AudioCLIP(pretrained=f'saved_ckpts/AudioCLIP-Full-Training.pt')

    def unscale_timestep(self, t):
        unscaled_timestep = (t * (self.diffusion.num_timesteps / 1000)).long()

        return unscaled_timestep

    def clip_loss(self, x_in, audio_embed):
        # x_in.shape(1,3,16,128,128)
        dists = 0

        if self.mask is not None:
            masked_input = x_in * self.mask
        else:
            masked_input = x_in
             
        if self.args.audio_emb_model in ['wav2clip', 'beats']:
            for idx in range(self.args.sequence_length):
                masked_frame = masked_input[:,:,idx,:,:]
                augmented_input = self.image_augmentations(masked_frame).add(1).div(2)
                clip_in = self.clip_normalize(augmented_input)
                image_embeds = self.clip_model.encode_image(clip_in).float()
                dist = d_clip_loss(image_embeds, audio_embed[idx,:], use_cosine=True) / self.args.sequence_length
                dists += dist.mean()
                
        elif self.args.audio_emb_model == 'audioclip':
            masked_frames = None        
            masked_input = masked_input.permute(0,2,1,3,4)
            masked_input = self.clip_normalize(masked_input)
            
            # resize every frame in a video. output shape: (1,16,3,224,224)
            for idx in range(masked_input.shape[1]):
                masked_frame = F.resize(masked_input[:,idx], [self.clip_size, self.clip_size])
                masked_frame = masked_frame.unsqueeze(1)
                if masked_frames == None:
                    masked_frames = masked_frame
                else:
                    masked_frames = torch.cat((masked_frames, masked_frame),1)
            
            # We want to sum over the averages, bs = 1, if change bs, need to modify the codes
            for bs in range(masked_frames.shape[0]):
                ((_, image_features, _), _), _ = self.audioclip(image=masked_frames[bs].cpu())       
                dist = audioclip_loss(audio_embed, image_features.cuda(), self.audioclip, use_scale=False) #(16,16)
                dist = torch.diag(dist).mean()
            
                dists += dist /  masked_frames.shape[0]

        return dists
    
    def direction_loss(self, x, embed):      
        dists = 0
        
        if self.args.audio_emb_model in ['wav2clip', 'beats']:
            for idx in range(self.args.sequence_length-1):
                x2 = F.resize(x[:,:,idx+1,:,:], [self.clip_size, self.clip_size])
                x1 = F.resize(x[:,:,idx,:,:], [self.clip_size, self.clip_size])
                dis_x = self.clip_model.encode_image(x2).float() - self.clip_model.encode_image(x1).float()
                dis_embed = embed[idx+1] - embed[idx]
                dist = d_clip_loss(dis_x, dis_embed, use_cosine=True)
                dists += dist / (self.args.sequence_length - 1)
        elif self.args.audio_emb_model == 'audioclip':  
            frames = None 
            x = x.permute(0,2,1,3,4)
            x = self.clip_normalize(x)
            
            # get resized frames (bs, frames, channel, resolution, resolution)
            for idx in range(x.shape[1]):
                frame = F.resize(x[:,idx], [self.clip_size, self.clip_size])
                frame = frame.unsqueeze(1)
                if frames == None:
                    frames = frame
                else:
                    frames = torch.cat((frames, frame),1)
                    
            # We want to sum over the averages, bs = 1, if change bs, need to modify the codes
            for bs in range(frames.shape[0]):   
                ((_, image_features, _), _), _ = self.audioclip(image=frames.squeeze().cpu()) #(16,1024)
                image_features = image_features.cuda()
                
                for idx in range(image_features.shape[0]-1):
                    dis_x = image_features[idx+1]-image_features[idx]
                    dis_embed = embed[idx+1] - embed[idx]               
                    dist = audioclip_loss(dis_embed, dis_x, self.audioclip, use_scale=False)
                    dists += dist / (image_features.shape[0] - 1)
        return dists

    def text_loss(self, x_in, text): 
        dists = 0
        frames = None 
        x_in = x_in.permute(0,2,1,3,4)
        x_in = self.clip_normalize(x_in).to(self.device)
        
        for idx in range(x_in.shape[1]):
            frame = F.resize(x_in[:,idx], [self.clip_size, self.clip_size])
            frame = frame.unsqueeze(1)
            if frames == None:
                frames = frame
            else:
                frames = torch.cat((frames, frame),1)
            
        for idx in range(self.args.sequence_length):
            tokenized_text = clip.tokenize(text).to(self.device)
            image_embeds = self.clip_model.encode_image(frames[:,idx,:,:,:]).float()
            text_embeds = self.clip_model.encode_text(tokenized_text).float()
            dist = d_clip_loss(image_embeds, text_embeds, use_cosine=True) / self.args.sequence_length
            dists += dist
        
        return dists
        
    def unaugmented_clip_distance(self, x, audio_embed):
        dists = 0
        if self.args.audio_emb_model in ['wav2clip', 'beats']:
            for idx in range(self.args.sequence_length):
                z = F.resize(x[:,:,idx,:,:], [self.clip_size, self.clip_size])
                image_embeds = self.clip_model.encode_image(z).float()
                dist = d_clip_loss(image_embeds, audio_embed[idx,:], use_cosine=True) / self.args.sequence_length
                dists += dist
        elif self.args.audio_emb_model == 'audioclip':
            frames = None
            
            x = x.permute(0,2,1,3,4)
            x = self.clip_normalize(x)
            
            for idx in range(x.shape[1]):
                frame = F.resize(x[:,idx], [self.clip_size, self.clip_size])
                frame = frame.unsqueeze(1)
                if frames == None:
                    frames = frame
                else:
                    frames = torch.cat((frames, frame),1)
            
            ((_, image_features, _), _), _ = self.audioclip(image=frames.squeeze().cpu())      
            dists = audioclip_loss(audio_embed, image_features.cuda(), self.audioclip, use_scale=True)
            dists = torch.diag(dists)

        return dists.mean().item()

    def edit_video_by_prompt(self, x, audio=None, raw_text=None, text=None):
        #text_embed = self.clip_model.encode_text(clip.tokenize(self.args.prompt).to(self.device)).float()
        if audio is not None:
            if self.args.audio_emb_model in ['wav2clip', 'beats']:
                audio_embed = torch.from_numpy(wav2clip.embed_audio(audio.cpu().numpy().squeeze(), self.wav2clip_model)).cuda() #(16,512)
            elif self.args.audio_emb_model == 'audioclip':
                ((audio_embed, _, _), _), _ = self.audioclip(audio=audio)
                audio_embed = audio_embed.cuda() #(16,1024)

       
        def cond_fn(x, t, c, y=None):
            if self.args.prompt == "":
                return torch.zeros_like(x)
            
            c=text

            with torch.enable_grad():
                x = x.detach().requires_grad_()
                t = self.unscale_timestep(t)

                out = self.diffusion.p_mean_variance(
                    self.model, x, t, c, clip_denoised=False, model_kwargs={"y": y}
                )

                fac = self.diffusion.sqrt_one_minus_alphas_cumprod[t[0].item()]
                x_in = out["pred_xstart"] * fac + x * (1 - fac)
                # x_in = out["pred_xstart"]

                loss = torch.tensor(0)
                if audio is not None:
                    if self.args.audio_guidance_lambda != 0:
                        clip_loss = self.clip_loss(x_in, audio_embed) * self.args.audio_guidance_lambda
                        loss = loss + clip_loss
                        self.metrics_accumulator.update_metric("clip_loss", clip_loss.item())

                    if self.args.direction_lambda != 0:
                        direction_loss = self.direction_loss(x_in, audio_embed) * self.args.direction_lambda
                        loss = loss + direction_loss
                        self.metrics_accumulator.update_metric("direction_loss", direction_loss.item())
                    
                if self.args.range_lambda != 0:
                    r_loss = range_loss(out["pred_xstart"]).sum() * self.args.range_lambda
                    loss = loss + r_loss
                    self.metrics_accumulator.update_metric("range_loss", r_loss.item())
                    
                if raw_text is not None:
                    if self.args.text_guidance_lambda != 0:
                        text_loss = self.text_loss(x_in, raw_text) * self.args.text_guidance_lambda
                        loss = loss + text_loss
                        self.metrics_accumulator.update_metric("text_loss", text_loss.item())

                return -torch.autograd.grad(loss, x)[0]

        @torch.no_grad()
        def postprocess_fn(out, t):
            if self.mask is not None:
                background_stage_t = self.diffusion.q_sample(self.init_image, t[0])
                background_stage_t = torch.tile(
                    background_stage_t, dims=(self.args.batch_size, 1, 1, 1)
                )
                out["sample"] = out["sample"] * self.mask + background_stage_t * (1 - self.mask)

            return out
        
        self.image_size = (self.model_config["image_size"], self.model_config["image_size"])
        self.init_image = x.to(self.device).unsqueeze(0)
        self.mask = torch.ones_like(self.init_image, device=self.device)
        save_image_interval = self.diffusion.num_timesteps // 5
        
        for iteration_number in range(self.args.iterations_num):
            print(f"Start iterations {iteration_number}")

            sample_func = (
                self.diffusion.ddim_sample_loop_progressive
                if self.args.ddim
                else self.diffusion.p_sample_loop_progressive
            )
            samples = sample_func(
                self.model,
                (
                    self.args.batch_size,
                    3,
                    16,
                    self.model_config["image_size"],
                    self.model_config["image_size"],
                ),
                text,
                clip_denoised=False,
                model_kwargs={}
                if self.args.model_output_size == 128
                else {
                    "y": torch.zeros([self.args.batch_size], device=self.device, dtype=torch.long)
                },
                cond_fn=None,
                progress=True,
                skip_timesteps=self.args.skip_timesteps,
                init_image=self.init_image,
                postprocess_fn=None if self.args.local_clip_guided_diffusion else postprocess_fn,
                randomize_class=True,
            )

            intermediate_samples = [[] for i in range(self.args.batch_size)]
            total_steps = self.diffusion.num_timesteps - self.args.skip_timesteps - 1

            for j, sample in enumerate(samples):
                should_save_image = j % save_image_interval == 0 or j == total_steps
                if should_save_image or self.args.save_video:
                    self.metrics_accumulator.print_average_metric()

                    for b in range(self.args.batch_size):
                        pred_image = sample["pred_xstart"][b]
                        
                        visualization_path = Path(
                            os.path.join(self.args.output_path, self.args.output_file)
                        )
                        visualization_path = visualization_path.with_name(
                            f"{visualization_path.stem}_i_{iteration_number}_b_{b}.mp4"
                        )

                        if (
                            self.mask is not None
                            and self.args.enforce_background
                            and j == total_steps
                            and not self.args.local_clip_guided_diffusion
                        ):
                            pred_image = (
                                self.init_image[0] * (1 - self.mask[0]) + pred_image * self.mask[0]
                            )
                        #pred_image = pred_image.add(1).div(2).clamp(0, 1)
                        print(pred_image.min(), pred_image.max())
                        pred_image = pred_image.clamp(-0.5,0.5)+0.5
                       
                        #final_distance = self.unaugmented_clip_distance(pred_image.unsqueeze(0), audio_embed)
                        #formatted_distance = f"{final_distance:.4f}"
                        #logger.log('final_distance: ', formatted_distance)
                        
            if self.args.save_video:
                for b in range(self.args.batch_size):
                    video_name = self.args.output_file.replace(
                        ".png", f"_i_{iteration_number}_b_{b}.mp4"
                    )
                    video_path = os.path.join(self.args.output_path, video_name)
                    save_video(intermediate_samples[b], video_path)
        
        return pred_image.unsqueeze(0)