

# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)

# SPDX-License-Identifier: MIT

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2022-11-15"


import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.colors as mcolors
import torch
from torch.optim import Adam
import torch.nn.functional as F
import sys
sys.path.append("..")
from utils.data_loader import create_loader
from models.network_unet import Upsampling
from models.network_diffusion import UNet
from torchvision.utils import save_image
from main_scripts.dataset_prep import PrecipDatasetInter

print(sys.version)
print(xr.__version__)

 # create colormaps
clevs = np.array([0, 1, 2.5, 5, 7.5, 10, 15, 20, 30, 40,
         50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 750])*1e-2
label = 'Precipitation Rate [mm/hour]'
cmap_data = [(1.0, 1.0, 1.0),
             (0.3137255012989044, 0.8156862854957581, 0.8156862854957581),
             (0.0, 1.0, 1.0),
             (0.0, 0.8784313797950745, 0.501960813999176),
             (0.0, 0.7529411911964417, 0.0),
             (0.501960813999176, 0.8784313797950745, 0.0),
             (1.0, 1.0, 0.0),
             (1.0, 0.6274510025978088, 0.0),
             (1.0, 0.0, 0.0),
             (1.0, 0.125490203499794, 0.501960813999176),
             (0.9411764740943909, 0.250980406999588, 1.0),
             (0.501960813999176, 0.125490203499794, 1.0),
             (0.250980406999588, 0.250980406999588, 1.0),
             (0.125490203499794, 0.125490203499794, 0.501960813999176),
             (0.125490203499794, 0.125490203499794, 0.125490203499794),
             (0.501960813999176, 0.501960813999176, 0.501960813999176),
             (0.8784313797950745, 0.8784313797950745, 0.8784313797950745),
             (0.9333333373069763, 0.8313725590705872, 0.7372549176216125),
             (0.8549019694328308, 0.6509804129600525, 0.47058823704719543),
             (0.6274510025978088, 0.42352941632270813, 0.23529411852359772),
             (0.4000000059604645, 0.20000000298023224, 0.0)]

cmap = mcolors.ListedColormap(cmap_data, 'precipitation')
cmap.set_bad(color='grey')
norm = mcolors.BoundaryNorm(clevs, cmap.N)

def spatial_plot(prcp, title, figname):
    fig = plt.figure(figsize = (8, 6))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    if '_in' in title:
        dx, dy = 0.1, 0.1
    else:
        dx, dy = 0.1, 0.1
    y, x = np.mgrid[slice(-8, 8 + dy, dy),
                    slice(-8, 8 + dx, dx)]
    # vmin = 0., vmax=1.0
    cs = plt.pcolormesh(x, y, prcp, cmap = cmap,vmin = -1, vmax=10.0)
    plt.yticks(np.arange(-8, 10, 2), fontsize = 18)
    plt.xticks(np.arange(-8, 10, 2), fontsize = 18)

    # # add colorbar.
    label = 'Precipitation rate [mm/h]'
    cbar = plt.colorbar(cs, location = 'right', pad = 0.03)
    cbar.set_label(label, fontsize = 20)  # 10$\mathregular{^-}$$\mathregular{^1}$

    plt.title(title, fontsize = 20, loc = 'center', pad = 6)

    # save to disk
    plt.savefig(figname, bbox_inches = "tight")
    plt.show()


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



######Forward difussion
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


timesteps = 200

# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps)


# define alphas
alphas = 1. - betas
#Returns the cumulative product of elements of input in the dimension dim
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)


# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)



# forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None):
    if not torch.is_tensor(x_start):
        x_start = torch.from_numpy(x_start)

    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    spls = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    return spls



def get_noisy_image(x_start, t):
  # add noise
  noisy_image = q_sample(x_start, t=t)
  # turn back into PIL image
  #noisy_image = reverse_transform(x_noisy.squeeze())
  return noisy_image


train_dir: str = "../../data/"
train_loader, batch_size = create_loader(train_dir)
upsampling = Upsampling(in_channels = 8)

# for i, train_data in enumerate(train_loader):
#     if i < 4:
#
#         x_up = upsampling(train_data["L"])
#         x = x_up.numpy()
#         print("shape of x", x.shape)
#         x1_start = x[0][-1] # The fist variable
#
#         #get noise output
#         x1 = [get_noisy_image(x1_start, torch.tensor([t])) for t in [1, 10, 30,100]]
#         x1 = [x.numpy() for x in x1]
#         print("x1_start",np.amin(x1_start))
#         spatial_plot(x1_start, "raw", "./")
#
#         print("x1", len(x1))
#         for j in range(len(x1)):
#            title = str(j) + "_precp_in"
#            spatial_plot(x1[j], title, "./")
#     else:
#         break



#define the loss function
def p_losses(denoise_model, x_start, t, noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss

@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)

    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise


def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs


def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

results_folder = Path("./results")
results_folder.mkdir(exist_ok = True)
save_and_sample_every = 1000


#training
device = "cuda" if torch.cuda.is_available() else "cpu"
model = UNet(n_channels=8)
model.to(device)
optimizer = Adam(model.parameters(), lr=1e-3)



epochs = 5
for epoch in range(epochs):
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        batch_size = batch["L"].shape[0]
        batch = upsampling(batch["L"])

        x1_start = batch[0][-1] # The fist variable

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()
        loss = p_losses(model, batch, t, loss_type="huber")
        if step % 10 == 0:
            print("Loss:", loss.item())

        loss.backward()
        optimizer.step()
        # save generated images
        if step != 0 and step % save_and_sample_every == 0:
            milestone = step // save_and_sample_every
            batches = num_to_groups(4, batch_size)
            all_images_list = list(map(lambda n: sample(model, batch_size=n, channels=8), batches))
            all_images = torch.cat(all_images_list, dim=0)
            all_images = (all_images + 1) * 0.5
            save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow = 6)