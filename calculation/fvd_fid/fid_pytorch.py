# Adapted from https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py

"""Calculates the Frechet Inception Distance (FID) to evalulate model

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import six
import os, sys, inspect
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.compat.v1 as tf
import tensorflow_gan as tfgan
import tensorflow_hub as hub
import torch
import argparse
from glob import glob
from joblib import Parallel, delayed
import numpy as np
import cv2
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
code_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, code_dir)

from calculation.utils import calculate_frechet_distance
from calculation.fvd_fid.inception import InceptionV3


def preprocess(videos, target_resolution):
    """Runs some preprocessing on the videos for I3D model.
    Args:
        videos: <T>[batch_size, num_frames, height, width, depth] The videos to be
            preprocessed. We don't care about the specific dtype of the videos, it can
            be anything that tf.image.resize_bilinear accepts. Values are expected to
            be in the range 0-255.
        target_resolution: (width, height): target video resolution
    Returns:
        videos: <float32>[batch_size, num_frames, height, width, depth] Values are in 
            the range [-1,1]
    """
    videos_shape = videos.shape.as_list()
    all_frames = tf.reshape(videos, [-1] + videos_shape[-3:])
    resized_videos = tf.image.resize_bilinear(all_frames, size=target_resolution)
    target_shape = [videos_shape[0], -1] + [3] + list(target_resolution) 
    output_videos = tf.reshape(resized_videos, target_shape)
    scaled_videos = 2. * tf.cast(output_videos, tf.float32) / 255. - 1
    return scaled_videos


def compute_fid_given_acts(acts_1, acts_2):
    """Computes the FVD of two paths"""
    m1 = np.mean(acts_1, axis=0)
    s1 = np.cov(acts_1, rowvar=False)
    m2 = np.mean(acts_2, axis=0)
    s2 = np.cov(acts_2, rowvar=False)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)
    return fid_value

def emb_from_files(real_video_files, fake_video_files, resize, num_workers):
    # both have dimensionality [NUMBER_OF_VIDEOS, VIDEO_LENGTH, FRAME_WIDTH, FRAME_HEIGHT, 3] with values in 0-255
    batch_size = 2
    total_size = len(real_video_files)
    
    real_embs = []
    fake_embs = []
    
    model = InceptionV3([3]).to("cuda")
    
    with tf.device('/device:GPU:0'):
        for i in tqdm(range(total_size // batch_size)):
            start = i * batch_size
            end = min(start + batch_size, total_size)
            real_videos = torch.from_numpy(load_videos([real_video_files[i] for i in range(start, end)], resize, num_workers)) #(1,16,112,112,3)
            real_videos = real_videos.view(-1, real_videos.shape[4], real_videos.shape[2], real_videos.shape[3])
            real_videos = real_videos / 255.
            real_videos = real_videos.to("cuda")
            real_emb = model(real_videos)[0]
            real_emb = real_emb.squeeze(3).squeeze(2).cpu().numpy()
                                               
            generated_videos = torch.from_numpy(load_videos([fake_video_files[i] for i in range(start, end)], resize, num_workers))
            generated_videos = generated_videos.view(-1, generated_videos.shape[4], generated_videos.shape[2], generated_videos.shape[3])
            generated_videos = generated_videos / 255.
            generated_videos = generated_videos.to("cuda")
            fake_emb = model(generated_videos)[0]
            fake_emb = fake_emb.squeeze(3).squeeze(2).cpu().numpy()
            
            real_embs.append(real_emb)
            fake_embs.append(fake_emb)

        real_embs = np.concatenate(real_embs, axis=0)
        fake_embs = np.concatenate(fake_embs, axis=0)
            
    return real_embs, fake_embs

def load_video(file, resize):
    vidcap = cv2.VideoCapture(file)
    success, image = vidcap.read()
    frames = []
    while success:
        if resize is not None:
            h, w = resize
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        frames.append(image)
        success, image = vidcap.read()
    return np.stack(frames)

def get_video_files(folder):
    return sorted(glob(os.path.join(folder, "*.mp4")))

def load_videos(video_files, resize, num_workers):
    videos = Parallel(n_jobs=num_workers)(delayed(load_video)(file, resize) for file in video_files)
    return np.stack(videos)

def get_folder(exp_tag, fold_i=None):
    if fold_i is not None:
        exp_tag = f"{fold_i}_" + exp_tag
    all_folders = glob(f"./results/*{exp_tag}")
    assert len(all_folders) == 1, f"Too many possibilities for this tag {exp_tag}:\n{all_folders}"
    return all_folders[0]

def get_folders(exp_tag, num_folds):
    if num_folds is not None:
        folders= []
        for i in range(num_folds):
            folders.append(get_folder(exp_tag, i))
        return folders
    else:
        return [get_folder(exp_tag)]

def fid_size(real_emb, fake_emb, size):
    fids = []
    print('fvd_size real_emb.shape: ', real_emb.shape)
    n = real_emb.shape[0] // size
    for i in tqdm(range(n)):
        r = real_emb[i * size:(i + 1) * size]
        f = fake_emb[i * size:(i + 1) * size]
        fids.append(compute_fid_given_acts(r, f))
    print("Individual FID scores")
    print(fids)
    print(f"Mean/std of FID across {n} runs of size {size}")
    print(np.mean(fids), np.around(np.std(fids), decimals=3))

def fid_full(real_emb, fake_emb):
    fid = compute_fid_given_acts(real_emb, fake_emb)
    print(f"FID score: {fid}")

def main(args):
    fake_folders = get_folders(args.exp_tag, args.num_folds)
    real_tag = args.exp_tag if args.real_tag is None else args.real_tag
    real_folders = get_folders(real_tag, args.num_folds)
    print('fake_folders: ', fake_folders)
    print('real_folders: ', real_folders)
    
    real_emb, fake_emb = [], []
    for i, (real_root, fake_root) in tqdm(enumerate(zip(sorted(real_folders), sorted(fake_folders)))):
        print(f"[{i}] Loading real")
        real_video_files = get_video_files(os.path.join(real_root, args.real_folder))
        print(f"Found {len(real_video_files)} {args.real_folder} video files")

        print(f"[{i}] Loading fake")
        fake_video_files = get_video_files(os.path.join(fake_root, args.fake_folder))
        print(f"Found {len(fake_video_files)} {args.fake_folder} video files")

        assert len(real_video_files) == len(fake_video_files)

        print(f"[{i}] Computing activations")
        real_emb_i, fake_emb_i = emb_from_files(real_video_files, fake_video_files, args.resize, args.num_workers)
        real_emb.append(real_emb_i)
        fake_emb.append(fake_emb_i)
        
    real_emb = np.concatenate(real_emb, axis=0)
    fake_emb = np.concatenate(fake_emb, axis=0)
    
    print(f"Computing FID with {args.mode} mode")
    if args.mode == "size" or args.mode == "both":
        fid_size(real_emb, fake_emb, args.size)
    if args.mode == "full" or args.mode == "both":
        fid_full(real_emb, fake_emb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_tag', type=str, default=None)
    parser.add_argument('--real_tag', type=str, default=None)
    parser.add_argument('--real_folder', type=str, default="real")
    parser.add_argument('--fake_folder', type=str, default="fake")
    parser.add_argument('--num_folds', type=int, default=None)
    parser.add_argument('--mode', type=str, default="full", help="(size | full | both)")
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--resize', type=int, nargs="+", default=None)
    parser.add_argument('--num_workers', type=int, default=8)
    args = parser.parse_args()
    main(args)
