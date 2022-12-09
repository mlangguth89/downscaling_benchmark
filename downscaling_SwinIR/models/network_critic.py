__email__ = "maximbr@post.bgu.ac.il"
__author__ = "Maxim Bragilovski"
__date__ = "2022-12-08"

import torch
import torch.nn as nn


class Conv_Block(nn.Module):

    def __init__(self, in_channels: int = None, out_channels: int = None,
                 kernel_size: int = 3, padding: str = "same", bias=True, stride=None):
        """
        The convolutional block consists of one convolutional layer, bach normalization and activation function
        :param in_channels : the number of input channels
        :param out_channels: the number of output channels
        :param kernel_size : the kernel size
        :param padding     : the techniques for padding, either 'same' or 'valid' or integer
        """
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), bias=bias, stride=(2, 2), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)


class Discriminator(torch.nn.Module):
    def __init__(self, shape, num_conv: int = 4, channels_start: int = 64, kernel: tuple = (3, 3),
                 stride: tuple = (2, 2), activation: str = "relu", lbatch_norm: bool = True):
        super().__init__()
        in_channels = shape[0]
        n_layers = nn.ModuleList([])

        out_channgels = channels_start
        for _ in range(num_conv):
            n_layers.append(Conv_Block(in_channels, out_channgels, kernel_size=kernel, stride=stride))
            in_channels = out_channgels
            out_channgels = out_channgels * 2

        self.conv_block_n = nn.Sequential(*n_layers)
        self.maxpool_conv = nn.AvgPool2d((6, 8))
        self.flat = nn.Flatten()
        self.hidden = nn.Linear(512, channels_start)
        self.output = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv_block_n(x)
        x = self.maxpool_conv(x)
        x = self.flat(x)
        x = self.hidden(x)
        return self.output(x)
