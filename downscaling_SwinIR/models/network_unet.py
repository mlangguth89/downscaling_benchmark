
# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT


__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2022-07-13"


import torch
import torch.nn as nn
import numpy as np
from torch import Tensor


class Upsampling(nn.Module):

    def __init__(self, in_channels:int = None, out_channels: int = None,
                 kernel_size: int = 3, padding: int = 1, stride: int = 2,
                 upsampling: bool = True, sf: int = 10, mode: str = "bilinear"):
        super().__init__()
        """
        This block is used for transposed low-resolution to the same dim as high-resolution before performing UNet
        Note: The input data is assumed to be of the form minibatch x channels x [optional depth] x [optional height] x width.
        :param in_channels : the number of input variables
        :param out_channels: the output channels for each ConvTranspose2D layer
        :param kernel_size : the kernel size
        :param padding     : the padding size
        :param stride      : the stride size
        :param sf          : the scaling factor (low-resolution to high resolution)
        :param upsampling  : use upsampling (True) to convert low to high resolution  or use transposed convolutional approach(False)
        """

        if upsampling:
            self.deconv_block = nn.Upsample(scale_factor = sf, mode = mode, align_corners = True)
        else:

            layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride,
                                         padding = padding) for i in range(3)]

            layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size = kernel_size, stride = 1,
                                   padding = padding, output_padding = 31 ))

            self.deconv_block = nn.Sequential(*layers)

    def forward(self, x:Tensor)->Tensor:
         return self.deconv_block(x)



class Conv_Block(nn.Module):

    def __init__(self, in_channels :int = None, out_channels: int = None,
                 kernel_size: int = 3, padding: str = "same",bias=True):
        """
        The convolutional block consists of one convolutional layer, bach normalization and activation function
        :param in_channels : the number of input channels
        :param out_channels: the number of output channels
        :param kernel_size : the kernel size
        :param padding     : the techniques for padding, either 'same' or 'valid' or integer
        """
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor)->Tensor: 
        return self.conv_block(x)



class Conv_Block_N(nn.Module):

    def __init__(self, in_channels:int = None, out_channels: int = None,
                 kernel_size: int = 3, padding: str = "same", n :int = 2):
        """

        :param in_channels : the number of input channels
        :param out_channels: the number of output channels
        :param kernel_size : the kernel size
        :param padding     : the techniques for padding, either 'same' or 'valid' or integer
        :param n           : the number of convolutional block
        """
        super().__init__()
        n_layers = [Conv_Block(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True)]
        for i in np.arange(n-1):
            print("i",i)
            n_layers.append(Conv_Block(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True))

        self.conv_block_n = nn.Sequential(*n_layers)

    def forward(self, x):
        return self.conv_block_n(x)


class Encoder_Block(nn.Module):
    """Downscaling with maxpool then convol block"""

    def __init__(self, in_channels: int = None, out_channels: int = None, kernel_maxpool: int = 2, l_large: bool = True):
        """
        One complete encoder-block used in U-net.
        :param in_channels   : the number of input channels
        :param out_channels  : the number of ouput channels
        :param kernel_maxpool: the number of kernel size
        :param l_large       : flag for large encoder (n consecutive convolutional block)
        """
        super().__init__()

        if l_large:
            self.layer1 = Conv_Block_N(in_channels, out_channels, n = 2)
        else:
            self.layer1 = Conv_Block_N(in_channels, out_channels)

        self.maxpool_conv = nn.MaxPool2d(kernel_maxpool)

    def forward(self, x:Tensor)->Tensor:
        x = self.layer1(x)
        e = self.maxpool_conv(x)
        return x, e


class Decode_Block(nn.Module):

    """Upscaling then double conv"""
    def __init__(self, in_channels: int = None, out_channels: int = None, kernel_size: int = 2,
                 stride_up: int = 2, padding: str = "same"):
        super().__init__()


        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride_up, padding = 1)
        self.conv = Conv_Block_N(in_channels, out_channels, n = 2, kernel_size = kernel_size, padding = padding)

    def forward(self, x1: Tensor, x2:Tensor)->Tensor:

        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)



class UNet(nn.Module):
    def __init__(self, n_channels, channels_start: int = 56):

        super(UNet, self).__init__()

        self.upsampling = Upsampling(n_channels, channels_start)

        """encoder """
        self.down1 = Encoder_Block(n_channels, channels_start)
        self.down2 = Encoder_Block(channels_start, channels_start*2)
        self.down3 = Encoder_Block(channels_start*2, channels_start*4)

        """ bridge encoder <-> decoder """
        self.b1 = Conv_Block(channels_start*4, channels_start*8)

        """decoder """
        self.up1 = Decode_Block(channels_start*8, channels_start*4)
        self.up2 = Decode_Block(channels_start*4, channels_start*2)
        self.up3 = Decode_Block(channels_start*2, channels_start)

        self.output = nn.Conv2d(channels_start, 1, kernel_size=1, bias=True)
        torch.nn.init.xavier_uniform(self.output .weight)


    def forward(self, x:Tensor)->Tensor:
        print("input shape",x.shape)
        x = self.upsampling(x)
        s1, e1 = self.down1(x)
        s2, e2 = self.down2(e1)
        s3, e3 = self.down3(e2)
        x4 = self.b1(e3)
        d1 = self.up1(x4, s3)
        d2 = self.up2(d1, s2)
        d3 = self.up3(d2, s1)
        output = self.output(d3)
        return output
