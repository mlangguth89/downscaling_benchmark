
# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)

# SPDX-License-Identifier: MIT

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2022-07-22"


import argparse
import sys
import os
import json
import torch
from collections import OrderedDict
from torch.optim import lr_scheduler
from torch.optim import Adam
import torch.nn as nn

sys.path.append('../')
from models.network_unet import UNet as unet
from models.network_vanilla_swin_transformer import SwinTransformerSR as swinSR
from models.network_swinir import SwinIR as swinIR
from models.network_vit import TransformerSR as vitSR
from models.network_swinunet_sys import SwinTransformerSys as swinUnet
from models.network_diffusion  import UNet_diff
from models.network_unet import Upsampling
from models.diffusion_utils import GaussianDiffusion
from models.network_critic import Discriminator as critic

from train_scripts.wgan_train import BuildWGANModel
from train_scripts.train import BuildModel
from utils.data_loader import create_loader
from flopth import flopth
###Weights and Bias
import wandb
os.environ["WANDB_MODE"]="offline"
##os.environ["WANDB_API_KEY"] = key
wandb.init(project="Precip_downscaling",reinit=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device",device)

available_models = ["unet", "wgan", "diffusion", "swinIR","swinUnet"]

def run(train_dir: str = "/p/scratch/deepacf/deeprain/bing/downscaling_maelstrom/train",
        test_dir: str = None,
        n_channels: int = 8,
        save_dir: str = "../results",
        checkpoint_save: int = 200,
        epochs: int = 2,
        type_net: str = "unet",
        batch_size: int = 4,
        patch_size: int = 2,
        window_size: int = 4,
        upscale_swinIR: int = 4,
        upsampler_swinIR: str = "pixelshuffle",
        dataset_type: str = "temperature",
        args: dict = None,
        **kwargs):

    """
    :param train_dir       : the directory that contains the training dataset NetCDF files
    :param test_dir        : the directory that contains the testing dataset NetCDF files
    :param checkpoint_save : how many steps to save checkpoint
    :param n_channels      : the number of input variables/channels
    :param save_dir        : the directory where the checkpoint results are save
    :param epochs          : the number of epochs
    :param type_net        : the type of the models
    """

    difussion = False
    conditional = None
    timesteps = None

    wandb.run.name = type_net
    if dataset_type=="precipitation":
        vars_in = ["cape_in", "tclw_in", "sp_in", "tcwv_in", "lsp_in", "cp_in", "tisr_in","u700_in","v700_in","yw_hourly_in"]
        vars_out = ["yw_hourly_tar"]
    elif dataset_type == "temperature":
        raise ("Not implement yet")

    train_loader = create_loader(file_path=train_dir, batch_size=batch_size, dataset_type=dataset_type, patch_size=16, k=0.01, mode="train")
    test_loader = create_loader(file_path=test_dir,
                               mode="test",
                               dataset_type=dataset_type,
                               k=0.01,
                               stat_path=train_dir,
                               patch_size=16,
                               verbose=1)
    print("The model {} is selected for training".format(type_net))
    if type_net == "unet":
        netG = unet(n_channels = n_channels,dataset_type=dataset_type)
    elif type_net == "swinIR":
        netG = swinIR(img_size=16,
                      patch_size=patch_size,
                      in_chans=n_channels,
                      window_size=window_size,
                      upscale=upscale_swinIR,
                      upsampler=upsampler_swinIR)
    elif type_net == "vitSR":
        netG = vitSR(embed_dim =768)
    elif type_net == "swinUnet":
        netG = swinUnet(img_size=160, patch_size=patch_size, in_chans=n_channels,
                        num_classes=1, embed_dim=96, depths=[2, 2, 2],
                        depths_decoder=[2, 2, 2], num_heads=[6, 6, 6],
                        window_size=window_size,
                        mlp_ratio=4, qkv_bias=True, qk_scale=None,
                        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                        ape=False,
                        final_upsample="expand_first")

    elif type_net == "diffusion":
        conditional = kwargs["conditional"]
        timesteps = kwargs["timesteps"]
        # add one channel for the noise
        netG = UNet_diff(img_size=160, n_channels=n_channels+1)
        difussion = True

    elif type_net == "wgan":
        netG = unet(n_channels=n_channels, dataset_type=dataset_type)
        netC = critic((1, 120, 96))

    else:
        raise NotImplementedError

    #calculate the model size
    #flops, params = flopth(netG, in_size = ((n_channels, 16, 16),))
    #print("flops, params", flops, params)


    #calculate the trainable parameters
    netG_params = sum(p.numel() for p in netG.parameters() if p.requires_grad)

    if type_net == "wgan":
        netC_params = sum(p.numel() for p in netC.parameters() if p.requires_grad)
        print("Total trainalbe parameters of the generator:", netG_params)
        print("Total trainalbe parameters of the critic:", netC_params)
    else:
        print("Total trainalbe parameters:", netG_params)


    if type_net == "wgan":
        model = BuildWGANModel(generator=netG, save_dir=save_dir, critic=netC, train_dataloader=train_loader,
                      val_dataloader=test_loader, checkpoint_save=checkpoint_save, hparams=args, dataset_type=dataset_type)
    else:
        model = BuildModel(netG, save_dir=save_dir, difussion=difussion, conditional=conditional, timesteps=timesteps, train_loader=train_loader, val_loader=test_loader, epochs=epochs,
                       checkpoint_save=checkpoint_save, dataset_type=dataset_type)

    wandb.config = {
        "lr": model.G_optimizer_lr,
        "train_dir": train_dir,
        "test_dir": test_dir,
        "epochs": epochs,
        "window_size": window_size,
        "patch_size": patch_size
    }

    model.fit()
                


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type = str, required = True,
                        help = "The directory where training data (.nc files) are stored")
    parser.add_argument("--test_dir", type = str, required = True,
                        help = "The directory where validation data (.nc files) are stored")
    parser.add_argument("--save_dir", type = str, help = "The checkpoint directory")
    parser.add_argument("--epochs", type = int, default = 2, help = "The checkpoint directory")
    parser.add_argument("--model_type", type = str, default = "unet", help = "The model type: unet, swinir")
    parser.add_argument("--dataset_type", type = str, help = "The dataset_type: precipitation, temperature")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--critic_iterations", type=float, default=5, help="The checkpoint directory")
    parser.add_argument("--lr_gn", type=float, default=1.e-05, help="The checkpoint directory")
    parser.add_argument("--lr_gn_end", type=float, default=1.e-06, help="The checkpoint directory")
    parser.add_argument("--lr_critic", type=float, default=1.e-06, help="The checkpoint directory")
    parser.add_argument("--decay_start", type=int, default=25, help="The checkpoint directory")
    parser.add_argument("--decay_end", type=int, default=50, help="The checkpoint directory")
    parser.add_argument("--lambada_gp", type=float, default=10, help="The checkpoint directory")
    parser.add_argument("--recon_weight", type=float, default=1000, help="The checkpoint directory")

    parser.add_argument("--dataset_type", type=str, default="precipitation",
                        help="The dataset type: temperature, precipitation")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--critic_iterations", type=float, default=5, help="The checkpoint directory")
    parser.add_argument("--lr_gn", type=float, default=1.e-05, help="The checkpoint directory")
    parser.add_argument("--lr_gn_end", type=float, default=1.e-06, help="The checkpoint directory")
    parser.add_argument("--lr_critic", type=float, default=1.e-06, help="The checkpoint directory")
    parser.add_argument("--decay_start", type=int, default=25, help="The checkpoint directory")
    parser.add_argument("--decay_end", type=int, default=50, help="The checkpoint directory")
    parser.add_argument("--lambada_gp", type=float, default=10, help="The checkpoint directory")
    parser.add_argument("--recon_weight", type=float, default=1000, help="The checkpoint directory")

    # PARAMETERS FOR SWIN-IR & SWIN-UNET
    parser.add_argument("--patch_size", type = int, default = 2)
    parser.add_argument("--window_size", type = int, default = 4)

    # PARAMETERS FOR SWIN-IR
    parser.add_argument("--upscale_swinIR", type = int, default = 4)
    parser.add_argument("--upsampler_swinIR", type = str, default = "pixelshuffle")

    #PARAMETERS FOR DIFFUSION
    parser.add_argument("--conditional", type = bool, default=True)
    parser.add_argument("--timesteps",type=int, default=200)
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    #save the args to the checkpoint directory
    with open(os.path.join(args.save_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(args), sort_keys = True, indent = 4))

    run(train_dir = args.train_dir,
<<<<<<< HEAD
        test_dir=args.test_dir,
        n_channels = 8,
=======
        val_dir = args.val_dir,
        n_channels = 10,
>>>>>>> 562e70e0b49f695bb4502a5277b371a02acf17bf
        save_dir = args.save_dir,
        checkpoint_save=10000,
        epochs = args.epochs,
        type_net = args.model_type,
        dataset_type = args.dataset_type,
        batch_size=args.batch_size,
        patch_size = args.patch_size,
        window_size = args.window_size,
        conditional=args.conditional,
        batch_size=args.batch_size,
        dataset_type=args.dataset_type,
        args=args)

if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    main()


