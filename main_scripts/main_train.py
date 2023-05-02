
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
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
available_models = ["unet", "wgan", "diffusion", "swinIR","swinUnet"]

def run(train_dir: str = "/p/scratch/deepacf/deeprain/bing/downscaling_maelstrom/train",
        val_dir: str = "/p/scratch/deepacf/deeprain/bing/downscaling_maelstrom/val",
        save_dir: str = "../results",
        checkpoint_save: int = 200,
        epochs: int = 2,
        type_net: str = "unet",
        dataset_type: str = "temperature",
        batch_size: int = 2,
        wandb_id = None,
        **kwargs):

    """
    :param train_dir       : the directory that contains the training dataset NetCDF files
    :param val_dir        : the directory that contains the testing dataset NetCDF files
    :param checkpoint_save : how many steps to save checkpoint
    :param n_channels      : the number of input variables/channels
    :param save_dir        : the directory where the checkpoint results are save
    :param epochs          : the number of epochs
    :param type_net        : the type of the models
    """
    wandb.init(project="Precip_downscaling",reinit=True ,id=wandb_id)
    #some parameters for diffusion models
    difussion = False
    conditional = None
    timesteps = None
    wandb.run.name = type_net
    type_net = type_net.lower() 

    # Set default hyper-parameters for all the models and used for wandb log
    config = {"epochs":epochs, "batch_size": batch_size, 
              "train_dir": train_dir, "val_dir":val_dir, 
              "save_dir":save_dir, "type_net":type_net,
              }
    # This is the model hparameters should be tailored to each model afterwards
    hparams = {}


    # Define parameters for datasets
    if dataset_type=="precipitation":
        n_channels = 8
        upscale=4
        img_size=[160, 160]
    elif dataset_type == "temperature":
        n_channels = 9
        upscale=1
        img_size=[96,120]
    
    # Load the datasets
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

    # Define the models
    if type_net == "unet":
        netG = unet(n_channels = n_channels,dataset_type=dataset_type)

    elif type_net == "swinir":
        netG = swinIR(img_size=img_size,
                      patch_size=4,
                      in_chans=n_channels,
                      window_size=2,
                      upscale=upscale,
                      upsampler= "pixelshuffle",
                      dataset_type=dataset_type)
    
    elif type_net.lower()  == "vitsr":
        netG = vitSR(embed_dim =768)

    elif type_net == "swinunet":
        netG = swinUnet(img_size=img_size, 
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
        netG = UNet_diff(img_size=img_size,
                         n_channels=n_channels+1)
        difussion = True

    elif type_net == "wgan":
        netG = unet(n_channels=n_channels, 
                    dataset_type=dataset_type)
        netC = critic((1, img_size[0], img_size[1]))

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


    # Build models for wgan and other models with one nerual network
    if type_net == "wgan":
        hparams = {"critic_iterations": 5, "lr_gn": 1.e-05, 
                  "lr_gn_end":1.e-06, "lr_critic": 1.e-06,
                  "decay_start":25, "decay_end": 50,
                  "lambada_gp":10, "recon_weight":1000} 
    
        
        model = BuildWGANModel(generator=netG,
                               save_dir=save_dir,
                               critic=netC,
                               train_loader=train_loader,
                               val_loader=val_loader,
                               hparams=config,
                               dataset_type=dataset_type)
    else:
        
        model = BuildModel(netG,
                           save_dir = save_dir,
                           difussion=difussion,
                           conditional=conditional,
                           timesteps=timesteps,
                           train_loader = train_loader,
                           val_loader = val_loader)
    config.update(hparams)
    wandb.config = config
    wandb.config.update({"lr":model.G_optimizer_lr})
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
    parser.add_argument("--wandb_id", type=str, default="None",help="wandb_id")

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
        batch_size=args.batch_size,
        dataset_type=args.dataset_type,
        wandb_id=args.wandb_id)

if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    main()

