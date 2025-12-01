"""
Visualize content and motion from vqgan.
"""
import sys
sys.path.append('/Users/zhaominglu/Desktop/program/TACM')
from tacm import VideoData
from tacm.download import load_vqgan
import os
import argparse
from matplotlib import pyplot
import torch.nn as nn


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class new_model(nn.Module):
    def __init__(self, model, output_layer=None):
        super().__init__()
        self.pretrained = model
        self.output_layer = output_layer

        self.net = nn.Sequential()
        for n, c in self.pretrained.named_children():
            self.net.add_module(n, c)
            #if n == "conv_blocks":
            #    self.net.add_module("conv_blocks1", c[0])
             #   self.net.add_module("conv_blocks2", c[1])
            #else:
              #  self.net.add_module(n, c)

            if n == self.output_layer:
                break


        self.pretrained = None

    def forward(self, x):
        x = self.net(x)
        return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = VideoData.add_data_specific_args(parser)
    parser.add_argument('--vqgan_ckpt', type=str)
    args = parser.parse_args()

    # load vqgan
    print("loading vqgan model...")
    first_stage_model = load_vqgan(args.vqgan_ckpt)
    for p in first_stage_model.parameters():
        p.requires_grad = False
    first_stage_model.codebook._need_init = False
    first_stage_model.eval()
    first_stage_model.train = disabled_train

    c_model = first_stage_model.content_encoder
    m_model = first_stage_model.motion_encoder

    # load data
    print("loading video data")
    data = VideoData(args)
    data = data.train_dataloader()
    batch = data.dataset.__getitem__(0)  # sample_id
    video = batch["video"].unsqueeze(0)

    # encode video
    #new_c_model = new_model(model=c_model, output_layer="conv_blocks")
    #print(new_c_model)
    #content = new_c_model(video)

    new_m_model = new_model(model=m_model, output_layer="conv_blocks")
    motion = new_m_model(video)

    #print("content.shape: ", content.shape)
    print("motion.shape: ", motion.shape)

    square = 8
    ix = 1
    for _ in range(square):
        for _ in range(square):
            # specify subplot and turn of axis
            ax = pyplot.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            pyplot.imshow(motion[0, ix-1, 0, :, :], cmap='gray')
            ix += 1
    # show the figure
    pyplot.show()