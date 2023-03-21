
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

sys.path.append('../')
from models.network_unet import UNet as unet
from models.network_swinir import SwinIR as swinIR
from models.network_vit import TransformerSR as vitSR
from models.network_swinunet_sys import SwinTransformerSys as swinUnet
from models.network_diffusion  import UNet_diff
from models.network_unet import Upsampling
from utils.data_loader import create_loader
from models.diffusion_utils import GaussianDiffusion
from models.network_critic import Discriminator as critic
from utils.data_loader import create_loader
from train_scripts.wgan_train import BuildWGANModel
from train_scripts.train import BuildModel
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
        val_dir: str = "/p/scratch/deepacf/deeprain/bing/downscaling_maelstrom/val",
        save_dir: str = "../results",
        checkpoint_save: int = 200,
        epochs: int = 2,
        type_net: str = "unet",
        dataset_type: str = "temperature",
        batch_size: int = 2,
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
        n_channels = 8
        upscale=4
    elif dataset_type == "temperature":
        n_channels = 9
        upscale=1
    train_loader = create_loader(train_dir, 
                                 batch_size = batch_size,
                                 patch_size=16, 
                                 stat_path=None,
                                 dataset_type = dataset_type)
    val_loader = create_loader(file_path=val_dir,
                               mode="test",
                               batch_size = batch_size,
                               stat_path=train_dir,
                               patch_size=16,
                               dataset_type=dataset_type)
    print("The model {} is selected for training".format(type_net))
    if type_net == "unet":
        netG = unet(n_channels = n_channels,dataset_type=dataset_type)
    elif type_net == "swinIR":
        netG = swinIR( dataset_type=dataset_type,
                     img_size=16,
                      patch_size=4,
                      in_chans=n_channels,
                      window_size=2,
                      upscale=upscale,
                      upsampler= "pixelshuffle")
    elif type_net == "vitSR":
        netG = vitSR(embed_dim =768)
    elif type_net == "swinUnet":
        netG = swinUnet(img_size=160, 
                        patch_size=4, 
                        in_chans=n_channels,
                        num_classes=1, 
                        embed_dim=96, 
                        depths=[2, 2, 2],
                        depths_decoder=[1,2, 2], 
                        num_heads=[6, 6, 6],
                        window_size=5,
                        mlp_ratio=4, 
                        qkv_bias=True, 
                        qk_scale=None,
                        drop_rate=0., 
                        attn_drop_rate=0., 
                        drop_path_rate=0.1,
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
        model = BuildWGANModel(generator=netG,
                               save_dir=save_dir,
                               critic=netC,
                               train_loader=train_loader,
                               val_loader=val_loader,
                               hparams=args,
                               dataset_type=dataset_type)
    else:
        model = BuildModel(netG,
                           save_dir = save_dir,
                           difussion=difussion,
                           conditional=conditional,
                           timesteps=timesteps,
                           train_loader = train_loader,
                           val_loader = val_loader
                           )

    wandb.config = {
        "lr": model.G_optimizer_lr,
        "train_dir": train_dir,
        "val_dir": val_dir,
        "epochs": epochs
    }


    model.fit()
                


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type = str, required = True,
                        help = "The directory where training data (.nc files) are stored")
    parser.add_argument("--val_dir", type = str, required = True,
                        help = "The directory where validation data (.nc files) are stored")
    parser.add_argument("--save_dir", type = str, help = "The checkpoint directory")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--epochs", type = int, default = 2, help = "The checkpoint directory")
    parser.add_argument("--model_type", type = str, default = "unet", help = "The model type: unet, swinir")
    parser.add_argument("--dataset_type", type=str, default="precipitation",
                        help="The dataset type: temperature, precipitation")
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
        val_dir = args.val_dir,
        save_dir = args.save_dir,
        epochs = args.epochs,
        type_net = args.model_type,
        patch_size = args.patch_size,
        conditional=args.conditional,
        batch_size=args.batch_size,
        dataset_type=args.dataset_type,
        args=args)

if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    main()
    # run(train_dir = "../../data/",
    #     val_dir = "../../data/",
    #     n_channels = 8,
    #     save_dir = '.',
    #     checkpoint_save = 20,
    #     epochs = 1,
    #     type_net = "difussion",
    #     conditional = True,
    #     timesteps=200
    #
    #     )
    #
