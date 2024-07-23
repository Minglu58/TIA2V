# adapted from https://github.com/guyyariv/TempoTokens/blob/master/av_align.py
"""
AV-Align Metric: Audio-Video Alignment Evaluation

AV-Align is a metric for evaluating the alignment between audio and video modalities in multimedia data.
It assesses synchronization by detecting audio and video peaks and calculating their Intersection over Union (IoU).
A higher IoU score indicates better alignment.

Usage:
- Provide a folder of video files as input.
- The script calculates the AV-Align score for the set of videos.
"""

import piq
import argparse
from glob import glob
from joblib import Parallel, delayed
import numpy as np
import cv2
from tqdm import tqdm
import os
import torch
#from torchmetrics.multimodal import CLIPScore
import torch.nn.functional as F
import clip
import torchvision.transforms.functional as T
import librosa
import librosa.display


# Function to extract frames from a video file
def extract_frames(video_path):
    """
    Extract frames from a video file.

    Args:
        video_path (str): Path to the input video file.

    Returns:
        frames (list): List of frames extracted from the video.
        frame_rate (float): Frame rate of the video.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    if not cap.isOpened():
        raise ValueError("Error: Unable to open the video file.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, frame_rate


# Function to detect audio peaks using the Onset Detection algorithm
def detect_audio_peaks(audio_file):
    """
    Detect audio peaks using the Onset Detection algorithm.

    Args:
        audio_file (str): Path to the audio file.

    Returns:
        onset_times (list): List of times (in seconds) where audio peaks occur.
    """
    y, sr = librosa.load(audio_file)
    # Calculate the onset envelope
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    # Get the onset events
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    return onset_times


# Function to find local maxima in a list
def find_local_max_indexes(arr, fps):
    """
    Find local maxima in a list.

    Args:
        arr (list): List of values to find local maxima in.
        fps (float): Frames per second, used to convert indexes to time.

    Returns:
        local_extrema_indexes (list): List of times (in seconds) where local maxima occur.
    """

    local_extrema_indexes = []
    n = len(arr)

    for i in range(1, n - 1):
        if arr[i - 1] < arr[i] > arr[i + 1]:  # Local maximum
            local_extrema_indexes.append(i / fps)

    return local_extrema_indexes


# Function to detect video peaks using Optical Flow
def detect_video_peaks(frames, fps):
    """
    Detect video peaks using Optical Flow.

    Args:
        frames (list): List of video frames.
        fps (float): Frame rate of the video.

    Returns:
        flow_trajectory (list): List of optical flow magnitudes for each frame.
        video_peaks (list): List of times (in seconds) where video peaks occur.
    """
    flow_trajectory = [compute_of(frames[0], frames[1])] + [compute_of(frames[i - 1], frames[i]) for i in range(1, len(frames))]
    video_peaks = find_local_max_indexes(flow_trajectory, fps)
    return flow_trajectory, video_peaks


# Function to compute the optical flow magnitude between two frames
def compute_of(img1, img2):
    """
    Compute the optical flow magnitude between two video frames.

    Args:
        img1 (numpy.ndarray): First video frame.
        img2 (numpy.ndarray): Second video frame.

    Returns:
        avg_magnitude (float): Average optical flow magnitude for the frame pair.
    """
    # Calculate the optical flow
    prev_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Calculate the magnitude of the optical flow vectors
    magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
    avg_magnitude = cv2.mean(magnitude)[0]
    return avg_magnitude


# Function to calculate Intersection over Union (IoU) for audio and video peaks
def calc_intersection_over_union(audio_peaks, video_peaks, fps):
    """
    Calculate Intersection over Union (IoU) between audio and video peaks.

    Args:
        audio_peaks (list): List of audio peak times (in seconds).
        video_peaks (list): List of video peak times (in seconds).
        fps (float): Frame rate of the video.

    Returns:
        iou (float): Intersection over Union score.
    """
    intersection_length = 0
    for audio_peak in audio_peaks:
        for video_peak in video_peaks:
            if video_peak - 1 / fps < audio_peak < video_peak + 1 / fps:
                intersection_length += 1
                break
    return intersection_length / (len(audio_peaks) + len(video_peaks) - intersection_length)


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
    return sorted(glob(os.path.join(folder, '*.mp4')))

def get_audio_files(folder):
    return sorted(glob(os.path.join(folder, '*.wav')))

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

def upscale(videos, min_size=96):
    h, w = videos.shape[-2:]
    if h >= min_size and w >= min_size:
        return videos
    else:
        if h < w:
            size = [min_size, int(min_size * w / h)]
        else:
            size = [int(min_size * h / w), min_size]
        return torch.nn.functional.interpolate(videos, size=size, mode='bilinear')

def print_scores(scores, name):
    print(f"Individual {name} scores")
    print(scores)
    print(f"Mean/std of {name} across {len(scores)} runs")
    print(np.mean(scores), np.around(np.std(scores), decimals=3))

def main(args):
    video_folders = get_folders(args.exp_tag, args.num_folds)
    audio_tag = args.exp_tag if args.audio_tag is None else args.audio_tag
    audio_folders = get_folders(audio_tag, args.num_folds)
    
    print('video_folders: ', video_folders)
    print('audio_folders: ', audio_folders)

    
    for i, (video_root, audio_root) in tqdm(enumerate(zip(sorted(video_folders), sorted(audio_folders)))):

        print(f"[{i}] Loading audio")
        audio_files = get_audio_files(os.path.join(audio_root, args.audio_folder))
        print(f"Found {len(audio_files)} {args.audio_folder} audio files")

        print(f"[{i}] Loading video")
        video_files = get_video_files(os.path.join(video_root, args.video_folder))
        print(f"Found {len(video_files)} {args.video_folder} video files")

        assert len(audio_files) == len(video_files)
        
        score = 0
        
        for audio_file, video_file in zip(sorted(audio_files), sorted(video_files)):
            frames, fps = extract_frames(video_file)

            audio_peaks = detect_audio_peaks(audio_file)
            flow_trajectory, video_peaks = detect_video_peaks(frames, fps)

            score += calc_intersection_over_union(audio_peaks, video_peaks, fps)

        print('AV-Align: ', score/len(video_files))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_tag', type=str, default=None)
    parser.add_argument('--audio_tag', type=str, default=None)
    parser.add_argument('--video_folder', type=str, default="fake")
    parser.add_argument('--audio_folder', type=str, default="audio")
    parser.add_argument('--num_folds', type=int, default=None)
    parser.add_argument('--sequence_length', type=int, default=16)
    parser.add_argument('--idx', type=int, nargs="+", default=[])
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--print_256', action='store_true')
    parser.add_argument('--resize', type=int, nargs="+", default=None)
    args = parser.parse_args()
    main(args)
