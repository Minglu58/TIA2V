# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import torch
import torchvision as tv
import imageio

import math
import numpy as np
import skvideo.io

import sys
import pdb as pdb_original

import tacm.modules.ignite_trainer as it

class ForkedPdb(pdb_original.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb_original.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin



# Shifts src_tf dim to dest dim
# i.e. shift_dim(x, 1, -1) would be (b, c, t, h, w) -> (b, t, h, w, c)
def shift_dim(x, src_dim=-1, dest_dim=-1, make_contiguous=True):
    n_dims = len(x.shape) # 5
    if src_dim < 0:
        src_dim = n_dims + src_dim
    if dest_dim < 0:
        dest_dim = n_dims + dest_dim # 4

    assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims

    dims = list(range(n_dims)) #[0,1,2,3,4]
    del dims[src_dim] #[0,2,3,4]

    permutation = []
    ctr = 0
    for i in range(n_dims): # [0,1,2,3,4]
        if i == dest_dim:
            permutation.append(src_dim)
        else:
            permutation.append(dims[ctr]) # permutation=[0,2,3,4,1]
            ctr += 1
    x = x.permute(permutation)
    if make_contiguous:
        x = x.contiguous()
    return x


# reshapes tensor start from dim i (inclusive)
# to dim j (exclusive) to the desired shape
# e.g. if x.shape = (b, thw, c) then
# view_range(x, 1, 2, (t, h, w)) returns
# x of shape (b, t, h, w, c)
def view_range(x, i, j, shape):
    shape = tuple(shape)

    n_dims = len(x.shape)
    if i < 0:
        i = n_dims + i

    if j is None:
        j = n_dims
    elif j < 0:
        j = n_dims + j

    assert 0 <= i < j <= n_dims

    x_shape = x.shape
    target_shape = x_shape[:i] + shape + x_shape[j:]
    return x.view(target_shape)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def tensor_slice(x, begin, size):
    assert all([b >= 0 for b in begin])
    size = [l - b if s == -1 else s
            for s, b, l in zip(size, begin, x.shape)]
    assert all([s >= 0 for s in size])

    slices = [slice(b, b + s) for b, s in zip(begin, size)]
    return x[slices]


def adopt_weight(global_step, threshold=0, value=0.):
    weight = 1
    if global_step < threshold:
        weight = value
    return weight


def save_video_grid(video, fname, nrow=None, fps=30):
    b, c, t, h, w = video.shape
    video = video.permute(0, 2, 3, 4, 1)
    video = (video.cpu().numpy() * 255).astype('uint8')
    if nrow is None:
        nrow = math.ceil(math.sqrt(b))
    ncol = math.ceil(b / nrow)
    padding = 1
    video_grid = np.zeros((t, (padding + h) * nrow + padding,
                           (padding + w) * ncol + padding, c), dtype='uint8')
    print(video_grid.shape)
    for i in range(b):
        r = i // ncol
        c = i % ncol
        start_r = (padding + h) * r
        start_c = (padding + w) * c
        video_grid[:, start_r:start_r + h, start_c:start_c + w] = video[i]
    video = []
    for i in range(t):
        video.append(video_grid[i])
    imageio.mimsave(fname, video, fps=fps)
    # skvideo.io.vwrite(fname, video_grid, inputdict={'-r': '5'})
    print('saved videos to', fname)


def comp_getattr(args, attr_name, default=None):
    if hasattr(args, attr_name):
        return getattr(args, attr_name)
    else:
        return default


def visualize_tensors(t, name=None, nest=0):
    if name is not None:
        print(name, "current nest: ", nest)
    print("type: ", type(t))
    if 'dict' in str(type(t)):
        print(t.keys())
        for k in t.keys():
            if t[k] is None:
                print(k, "None")
            else:
                if 'Tensor' in str(type(t[k])):
                    print(k, t[k].shape)
                elif 'dict' in str(type(t[k])):
                    print(k, 'dict')
                    visualize_tensors(t[k], name, nest + 1)
                elif 'list' in str(type(t[k])):
                    print(k, len(t[k]))
                    visualize_tensors(t[k], name, nest + 1)
    elif 'list' in str(type(t)):
        print("list length: ", len(t))
        for t2 in t:
            visualize_tensors(t2, name, nest + 1)
    elif 'Tensor' in str(type(t)):
        print(t.shape)
    else:
        print(t)
    return ""


## copy from audioclip/utils/transforms.py
def scale(old_value, old_min, old_max, new_min, new_max):
    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    new_value = (((old_value - old_min) * new_range) / old_range) + new_min

    return new_value


def frame_signal(signal: torch.Tensor,
                 frame_length: int,
                 hop_length: int,
                 window: torch.Tensor = None) -> torch.Tensor:

    if window is None:
        window = torch.ones(frame_length, dtype=signal.dtype, device=signal.device)

    if window.shape[0] != frame_length:
        raise ValueError('Wrong `window` length: expected {}, got {}'.format(window.shape[0], frame_length))

    signal_length = signal.shape[-1]

    if signal_length <= frame_length:
        num_frames = 1
    else:
        num_frames = 1 + int(math.ceil((1.0 * signal_length - frame_length) / hop_length))

    pad_len = int((num_frames - 1) * hop_length + frame_length)
    if pad_len > signal_length:
        zeros = torch.zeros(pad_len - signal_length, device=signal.device, dtype=signal.dtype)

        while zeros.dim() < signal.dim():
            zeros.unsqueeze_(0)

        pad_signal = torch.cat((zeros.expand(*signal.shape[:-1], -1)[..., :zeros.shape[-1] // 2], signal), dim=-1)
        pad_signal = torch.cat((pad_signal, zeros.expand(*signal.shape[:-1], -1)[..., zeros.shape[-1] // 2:]), dim=-1)
    else:
        pad_signal = signal

    indices = torch.arange(0, frame_length, device=signal.device).repeat(num_frames, 1)
    indices += torch.arange(
        0,
        num_frames * hop_length,
        hop_length,
        device=signal.device
    ).repeat(frame_length, 1).t_()
    indices = indices.long()

    frames = pad_signal[..., indices]
    frames = frames * window

    return frames


class ToTensor1D(tv.transforms.ToTensor):

    def __call__(self, tensor: np.ndarray):
        tensor_2d = super(ToTensor1D, self).__call__(tensor[..., np.newaxis])

        return tensor_2d.squeeze_(0)


class RandomFlip(it.AbstractTransform):

    def __init__(self, p: float = 0.5):
        super(RandomFlip, self).__init__()

        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            flip_mask = torch.rand(x.shape[0], device=x.device) <= self.p
            x[flip_mask] = x[flip_mask].flip(-1)
        else:
            if torch.rand(1) <= self.p:
                x = x.flip(0)

        return x


class RandomScale(it.AbstractTransform):

    def __init__(self, max_scale: float = 1.25):
        super(RandomScale, self).__init__()

        self.max_scale = max_scale

    @staticmethod
    def random_scale(max_scale: float, signal: torch.Tensor) -> torch.Tensor:
        scaling = np.power(max_scale, np.random.uniform(-1, 1))
        output_size = int(signal.shape[-1] * scaling)
        ref = torch.arange(output_size, device=signal.device, dtype=signal.dtype).div_(scaling)

        ref1 = ref.clone().type(torch.int64)
        ref2 = torch.min(ref1 + 1, torch.full_like(ref1, signal.shape[-1] - 1, dtype=torch.int64))
        r = ref - ref1.type(ref.type())
        scaled_signal = signal[..., ref1] * (1 - r) + signal[..., ref2] * r

        return scaled_signal

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.random_scale(self.max_scale, x)


class RandomCrop(it.AbstractTransform):

    def __init__(self, out_len: int = 44100, train: bool = True):
        super(RandomCrop, self).__init__()

        self.out_len = out_len
        self.train = train

    def random_crop(self, signal: torch.Tensor) -> torch.Tensor:
        if self.train:
            left = np.random.randint(0, signal.shape[-1] - self.out_len)
        else:
            left = int(round(0.5 * (signal.shape[-1] - self.out_len)))

        orig_std = signal.float().std() * 0.5
        output = signal[..., left:left + self.out_len]

        out_std = output.float().std()
        if out_std < orig_std:
            output = signal[..., :self.out_len]

        new_out_std = output.float().std()
        if orig_std > new_out_std > out_std:
            output = signal[..., -self.out_len:]

        return output

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.random_crop(x) if x.shape[-1] > self.out_len else x


class RandomPadding(it.AbstractTransform):

    def __init__(self, out_len: int = 88200, train: bool = True):
        super(RandomPadding, self).__init__()

        self.out_len = out_len
        self.train = train

    def random_pad(self, signal: torch.Tensor) -> torch.Tensor:
        if self.train:
            left = np.random.randint(0, self.out_len - signal.shape[-1])
        else:
            left = int(round(0.5 * (self.out_len - signal.shape[-1])))

        right = self.out_len - (left + signal.shape[-1])

        pad_value_left = signal[..., 0].float().mean().to(signal.dtype)
        pad_value_right = signal[..., -1].float().mean().to(signal.dtype)
        output = torch.cat((
            torch.zeros(signal.shape[:-1] + (left,), dtype=signal.dtype, device=signal.device).fill_(pad_value_left),
            signal,
            torch.zeros(signal.shape[:-1] + (right,), dtype=signal.dtype, device=signal.device).fill_(pad_value_right)
        ), dim=-1)

        return output

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.random_pad(x) if x.shape[-1] < self.out_len else x


class RandomNoise(it.AbstractTransform):

    def __init__(self, snr_min_db: float = -10.0, snr_max_db: float = 100.0, p: float = 0.5):
        super(RandomNoise, self).__init__()

        self.p = p
        self.snr_min_db = snr_min_db
        self.snr_max_db = snr_max_db

    def random_noise(self, signal: torch.Tensor) -> torch.Tensor:
        target_snr = np.random.rand() * (self.snr_max_db - self.snr_min_db + 1.0) + self.snr_min_db

        signal_watts = torch.mean(signal ** 2, dim=(-1, -2))
        signal_db = 10 * torch.log10(signal_watts)

        noise_db = signal_db - target_snr
        noise_watts = 10 ** (noise_db / 10)
        noise = torch.normal(0.0, noise_watts.item() ** 0.5, signal.shape)

        output = signal + noise

        return output

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.random_noise(x) if np.random.rand() <= self.p else x


