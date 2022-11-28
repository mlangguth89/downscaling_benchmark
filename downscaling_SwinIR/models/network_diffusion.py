# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT


__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2022-11-06"


import math
import torch
from torch import nn,Tensor
from einops import rearrange

def exists(x):
    return x is not None

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


### Building blocks
class Conv_Block(nn.Module):

    def __init__(self, in_channels:int = None, out_channels: int = None,
                 kernel_size: int = 3, bias=True, time_emb_dim=None):
        """
        The convolutional block consists of one convolutional layer, bach normalization and activation function
        :param in_channels : the number of input channels
        :param out_channels: the number of output channels
        :param kernel_size : the kernel size
        :param padding     : the techniques for padding, either 'same' or 'valid' or integer
        """
        super().__init__()

        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, in_channels))
        )

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding="same", bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor, time_emb:Tensor=None)->Tensor:

        condition = self.mlp(time_emb)
        condition = rearrange(condition, "b c -> b c 1 1")
        x = x + condition

        return self.conv_block(x)


class Conv_Block_N(nn.Module):

    def __init__(self, in_channels:int = None, out_channels: int = None,
                 kernel_size: int = 3, time_emb_dim: int=None):
        """

        :param in_channels : the number of input channels
        :param out_channels: the number of output channels
        :param kernel_size : the kernel size
        :param padding     : the techniques for padding, either 'same' or 'valid' or integer
        :param n           : the number of convolutional block
        """
        super().__init__()
        self.block1 = Conv_Block(in_channels, out_channels, kernel_size=kernel_size,
                               bias=True, time_emb_dim = time_emb_dim)

        self.block2 = Conv_Block(out_channels, out_channels, kernel_size=kernel_size,
                               bias=True, time_emb_dim = time_emb_dim)


    def forward(self, x, time_emb: Tensor = None):
        h = self.block1(x, time_emb)
        h = self.block2(h, time_emb)
        return h



class Encoder_Block(nn.Module):
    """Downscaling with maxpool then convol block"""

    def __init__(self, in_channels: int = None, out_channels: int = None, 
                 kernel_maxpool: int = 2, time_emb_dim=None):
        """
        One complete encoder-block used in U-net.
        :param in_channels   : the number of input channels
        :param out_channels  : the number of ouput channels
        :param kernel_maxpool: the number of kernel size
        :param l_large       : flag for large encoder (n consecutive convolutional block)
        """
        super().__init__()

        self.layer1 = Conv_Block_N(in_channels, out_channels, time_emb_dim=time_emb_dim)

        self.maxpool_conv = nn.MaxPool2d(kernel_maxpool)

    def forward(self, x:Tensor, time_emb: Tensor=None)->Tensor:
        x = self.layer1(x, time_emb)
        e = self.maxpool_conv(x)
        return x, e



class Decode_Block(nn.Module):

    """Upscaling then double conv"""
    def __init__(self, in_channels: int=None, out_channels: int = None, kernel_size: int = 2,
                 stride_up: int = 2,  time_emb_dim=None):

        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = kernel_size,
                                     stride=stride_up, padding = 0)
        self.conv = Conv_Block_N(in_channels, out_channels, kernel_size = kernel_size,
                                time_emb_dim = time_emb_dim)

    def forward(self, x1: Tensor, x2:Tensor, time_emb:Tensor = None)->Tensor:

        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x, time_emb)


#########Define Unet neural network for diffusion models ##############

class UNet(nn.Module):
    def __init__(self, n_channels, channels_start: int = 56, with_time_emb: bool=True):

        super(UNet, self).__init__()
        # time embeddings
        # The time embedding dims should be the same as the model dims in order to sum up of the two
        if with_time_emb:
            time_dim = n_channels * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(n_channels),
                nn.Linear(n_channels, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None
        

        """encoder """
        self.down1 = Encoder_Block(in_channels = n_channels, out_channels = channels_start, time_emb_dim = time_dim)
        self.down2 = Encoder_Block(in_channels = channels_start, out_channels = channels_start*2, time_emb_dim = time_dim)
        self.down3 = Encoder_Block(in_channels = channels_start*2, out_channels = channels_start*4, time_emb_dim = time_dim)

        """ bridge encoder <-> decoder """
        self.b1 = Conv_Block(channels_start*4, channels_start*8, time_emb_dim = time_dim)

        """decoder """
        self.up1 = Decode_Block(in_channels = channels_start*8, out_channels = channels_start*4, time_emb_dim = time_dim)
        self.up2 = Decode_Block(in_channels = channels_start*4, out_channels = channels_start*2, time_emb_dim = time_dim)
        self.up3 = Decode_Block(in_channels = channels_start*2, out_channels = channels_start, time_emb_dim = time_dim)

        self.output = nn.Conv2d(channels_start, 1, kernel_size=1, bias=True)
        torch.nn.init.xavier_uniform(self.output .weight)


    def forward(self, x:Tensor,time: Tensor=None)->Tensor:


        t = self.time_mlp(time) if exists(self.time_mlp) else None

        s1, e1 = self.down1(x, t)
        s2, e2 = self.down2(e1, t)
        s3, e3 = self.down3(e2, t)
        x4 = self.b1(e3, t)
        d1 = self.up1(x4, s3, t)
        d2 = self.up2(d1, s2, t)
        d3 = self.up3(d2, s1, t)
        output = self.output(d3)
        return output



