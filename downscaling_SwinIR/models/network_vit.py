
# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT


__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2022-07-20"

"""
The implementation is based on the tutorial from the following link
https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632
"""

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce
from einops import rearrange, reduce, repeat

class PatchEmbedding(nn.Module):

    def __init__(self,
                 in_channels: int = 8,
                 patch_size: int = 4,
                 emb_size: int = 64,
                 enable_cnn: bool = False,
                 img_size: int = 16):
        super().__init__()
        """
        in_channels :  the number of variables/channles
        patch_size  :  the size of each patch
        emb_size    :  the embedding size
        enable_cnn  :  if use convolutional network as projection, if false, using linear projection
        """
        self.patch_size = patch_size
        if enable_cnn:
            self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e')
            )

        else:
            self.projection = nn.Sequential(
            # break-down the image in s1 x s2 patches and flat them
            Rearrange('b c (h s1) (w s2) -> b (h w) (s1 s2 c)', s1=patch_size, s2=patch_size),
            nn.Linear(patch_size * patch_size * in_channels, emb_size)
            )
            # I remove the cls embedding, since we do not have class labels in the data,
            # I used learnable position embedding in this case
            # In the future, we can include the datetime and location as embedding, it should be implemented here
            self.position = nn.Parameter(torch.randn((img_size // patch_size) ** 2 , emb_size))
            print("self.position", self.position.shape)
    
    
    def forward(self, x: Tensor) ->Tensor:
        x = self.projection(x)
        print("The shape after Patching Embedding",x.shape)
        # add position embedding
        x += self.position
        return x



class MultiHeadAttention(nn.Module):
    def __init__(self,
                 emb_size: int = 768,
                 num_heads: int = 8,
                 dropout: float = 0):
        super().__init__()

        self.emb_size = emb_size
        self.num_heads = num_heads
        # fuse the queries, keys and values in one matrix
        self.qkv = nn.Linear(emb_size, emb_size * 3)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)


    def forward(self, x:Tensor, mask:Tensor=None)->Tensor:

        qkv = rearrange(self.qkv(x), "b n (h d qkv) -> (qkv) b h n d", h=self.num_heads, qkv=3)
        queries, keys, values = qkv[0], qkv[1], qkv[2]
        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)# batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy, dim = -1) / scaling
        att = self.att_drop(att)
        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int,
                 expansion: int = 4,
                 drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size: int = 768,
                 drop_p: float = 0.,
                 forward_expansion: int = 4,
                 forward_drop_p: float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, **kwargs),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class Upsample(nn.Sequential):
    """Upsample module.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale: int=10):
        m = []
        #m.append(nn.Conv2d(in_channels=num_feat, out_channels=scale*scale*num_feat, kernel_size=3, stride=1, padding=1))
        m.append(nn.PixelShuffle(scale))
        super(Upsample, self).__init__(*m)



class TransformerSR(nn.Module):

    def __init__(self, embed_dim: int = 64 , num_feat: int=1, 
            img_size = 16, upscale: int=10, patch_size:int=4, in_channels: int=8,out_channels:int=1):
        print("Transformer Build")
        super(TransformerSR, self).__init__()
        print("Building TransformerSR")
        self.patch_size = patch_size
        self.img_size = img_size
        self.upscale = upscale
        self.out_channels = out_channels
        
        self.embed = PatchEmbedding(in_channels, patch_size, embed_dim, img_size=img_size)
        self.TransformerEncode = TransformerEncoder(depth=2)
        # for SR
        #self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
        #                                             nn.LeakyReLU(inplace = True))

        self.linear = nn.Linear(embed_dim, patch_size*patch_size*out_channels*upscale*upscale) 
        self.upsample = Upsample(scale=upscale)
        self.conv_last = nn.Conv2d(num_feat, 1, 3, 1, 1)


    def forward(self,x):
        x = self.embed(x)
        x = self.TransformerEncode(x)
        print("x shape after TransformerEncode:",x.shape)
        x_shape = x.size()
        #x = x.permute(0, 2, 1) #put channle to   the second place
        print("X shape before conv_before_upsample",x.shape)
        x = self.linear(x)
        x = x.permute(0, 2, 1) #put channle to the second place
        print("x shape after permute",x.shape)
        x =  x.reshape(x.size(dim=0), self.out_channels*self.upscale*self.upscale, self.img_size, self.img_size)
        x = self.upsample(x)
        print("X shape after upsample",x.shape)

        x= self.conv_last(x)
        return x




