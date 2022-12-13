# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)

# SPDX-License-Identifier: MIT

"""
This implementation is based on two reference:
1) https://huggingface.co/blog/annotated-diffusion
2) https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/blob/master/model/ddpm_modules/diffusion.py
1) is a simple, unconditional diffussion, while 2) is conditional diffusion
"""

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2022-11-28"


import torch
import torch.nn.functional as F
from torch import nn, einsum
import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"


#############Define the schedules for T timestamps
def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    a = a.to(device)
    out = a.gather(-1, t.to(device))
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(device)


class GaussianDiffusion(nn.Module):
    def __init__(self, conditional=False, schedule_opt="linear", timesteps=200, model=None):
        super().__init__()
        self.conditional = conditional
        self.model = model

        # define beta schedule
        if schedule_opt == "linear":
            self.betas = linear_beta_schedule(timesteps)
        elif schedule_opt == "quad":
            self.betas = quadratic_beta_schedule(timesteps)
        elif schedule_opt == "cosine":
            self.betas = cosine_beta_schedule(timesteps)
        elif schedule_opt == "sigmoid":
            self.betas = sigmoid_beta_schedule(timesteps)
        else:
            raise NotImplementedError
        self.set_new_noise_schedule()

    def set_new_noise_schedule(self):
        alphas = 1. - self.betas
        # Returns the cumulative product of elements of input in the dimension dim
        alphas_cumprod = torch.cumprod(alphas, axis = 0)
        self.alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.0)
        #sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - alphas_cumprod)



    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise



    @torch.no_grad()
    def p_sample(self, x, t, t_index, condition_x=None):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        if condition_x is not None:
            # this is conditional diffussion
            x_recon = self.model(torch.cat([condition_x, x], dim=1), t)
        else:
            x_recon = self.model(x, t)
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * x_recon / sqrt_one_minus_alphas_cumprod_t
        )
        posterior_variance_t = extract(self.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # No noise when t == 0
        if t_index == 0:
            return model_mean
        else:
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise


    @torch.no_grad()
    def p_sample_loop(self, shape, x_in):
        #device = next(self.model.parameters()).device

        b = shape[0]
        img = torch.randn(shape, device = device)
        # start from pure noise (for each example in the batch)
        imgs = []
        #The following code get from reference 2). However, this is some difference.
        #Need to furthe check. i do not understand why ret_img part
        if not self.conditional:
            for i in tqdm(reversed(range(0, self.timesteps)),
                          desc = 'sampling loop time step',
                          total = self.timesteps):
                img = self.p_sample(img, torch.full((b,), i, device = device, dtype = torch.long), i)
                imgs.append(img.numpy())
            return imgs
        else:

            for i in tqdm(reversed(range(0, self.timesteps)),
                          desc = 'sampling loop time step',
                          total = self.timesteps):
                img = self.p_sample(img, torch.full((b,), i, device = device, dtype = torch.long), i, condition_x = x_in)
                imgs.append(img.numpy())
            return imgs

    def sample(self, image_size=160, batch_size=16, channels=3, x_in=None):
        if self.conditional:
            return self.p_sample_loop(shape=(batch_size, channels, image_size, image_size), x_in=x_in)
        else:
            return self.p_sample_loop(shape=(batch_size, channels, image_size, image_size))