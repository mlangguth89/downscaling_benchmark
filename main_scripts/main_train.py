
# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)

# SPDX-License-Identifier: MIT

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2022-07-22"

import time
import argparse
import sys
import os
import json
import torch
from collections import OrderedDict
from torch.optim import lr_scheduler
from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn
sys.path.append('../')
from models.network_unet import UNet as unet
from models.network_swinir import SwinIR as swinIR
#from models.network_vanilla_swin_transformer import SwinTransformerSR as swinSR
from models.network_vit import TransformerSR as vitSR
from models.network_swinunet_sys import SwinTransformerSys as swinUnet
from models.network_diffusion  import UNet_diff
from models.network_unet import Upsampling
from utils.data_loader import create_loader
from models.diffusion_utils import GaussianDiffusion
from flopth import flopth
###Weights and Bias
import wandb
os.environ["WANDB_MODE"]="offline"
##os.environ["WANDB_API_KEY"] = key
wandb.init(project="Precip_downscaling",reinit=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device",device)

class BuildModel:
    def __init__(self, netG, G_lossfn_type: str = "l2",
                 G_optimizer_type: str = "adam",
                 G_optimizer_lr: float = 2e-1,
                 G_optimizer_betas: list = [0.9, 0.999],
                 G_optimizer_wd: int= 0, save_dir: str = "../results",
                 difussion: bool=False, **kwargs):
        """

        :param netG:
        :param G_lossfn_type:
        :param G_optimizer_type:
        :param G_optimizer_lr:
        :param save_dir: str, the save model path
        :param difussion: if enable difussion, the conditional must be defined
        :param kwargs: conditional: bool
        """

        # ------------------------------------
        # define network
        # ------------------------------------
        self.netG = netG

        self.netG.to(device)
        self.G_lossfn_type = G_lossfn_type
        self.G_optimizer_type = G_optimizer_type
        self.G_optimizer_lr = G_optimizer_lr
        self.G_optimizer_betas = G_optimizer_betas
        self.G_optimizer_wd = G_optimizer_wd
        self.schedulers = []
        self.difussion = difussion
        self.save_dir = save_dir
        if difussion:
            self.conditional = kwargs["conditional"]
            self.timesteps = kwargs["timesteps"]


    def init_train(self):
        wandb.watch(self.netG, log_freq=100)
        self.netG.train()
        self.define_loss()
        self.define_optimizer()
        self.define_scheduler()
    # ----------------------------------------
    # define loss
    # ----------------------------------------
    def define_loss(self):

        if self.G_lossfn_type == 'l1':
            self.G_lossfn = nn.L1Loss()
        elif self.G_lossfn_type == 'l2':
            self.G_lossfn = nn.MSELoss()
        elif self.G_lossfn_type == "huber":
            self.G_lossfn = nn.SmoothL1Loss() ##need to check if this works or not
        else:
            raise NotImplementedError('Loss type [{:s}] is not found.'.format(self.G_lossfn_type))

    # ----------------------------------------
    # define optimizer
    # ----------------------------------------
    def define_optimizer(self):
        G_optim_params = []
        for k, v in self.netG.named_parameters():
            if v.requires_grad:
                G_optim_params.append(v)
            else:
                print('Params [{:s}] will not optimize.'.format(k))

        self.G_optimizer = Adam(G_optim_params, lr = self.G_optimizer_lr,
                                betas = self.G_optimizer_betas,
                                weight_decay = self.G_optimizer_wd)

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                        milestones = [4000, 20000, 50000],
                                                        gamma = 0.01))

    # ----------------------------------------
    # save model / optimizer(optional)
    # ----------------------------------------
    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)

    # ----------------------------------------
    # save the state_dict of the network
    # ----------------------------------------
    def save_network(self, save_dir, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    # ----------------------------------------
    # feed L/H data
    # ----------------------------------------
    def feed_data(self, data):
        #print("datat[L] shape",data["L"].shape)
        self.L = data['L']
        if self.difussion:
            upsampling = Upsampling(in_channels = 8)
            self.L = upsampling(self.L)
        self.H = data['H']

    def count_flops(self,data):
        # Count the number of FLOPs
        c_ops = count_ops(self.netG,data)
        print("The number of FLOPS is:",c_ops )

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self):
        #print('self.H shape: {}'.format(self.H.shape))
        #print('self.netG(self.L) shape: {}'.format(self.netG(self.L).shape))
        if not self.difussion:
            self.E = self.netG(self.L) #[:,0,:,:]
        else:
            if len(self.H.shape) == 3:
                self.H = torch.unsqueeze(self.H, dim = 1)
            h_shape = self.H.shape

            noise = torch.randn_like(self.H)
            t = torch.randint(0, self.timesteps, (h_shape[0],), device = device).long()
            gd = GaussianDiffusion(model = self.netG, timesteps = self.timesteps)
            x_noisy = gd.q_sample(x_start = self.H, t = t, noise = noise)

            if not self.conditional:
                self.E = self.netG(x_noisy, t)

            else:
                self.E = self.netG(torch.cat([self.L, x_noisy], dim = 1), t)

            self.H = noise #if using difussion, the output is not the prediction values, but the predicted noise

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self):
        self.G_optimizer.zero_grad()
        self.netG_forward()
        self.G_loss = self.G_lossfn(self.E, self.H)
        self.G_loss.backward()
        self.G_optimizer.step()

    def update_learning_rate(self, n):
        for scheduler in self.schedulers:
            scheduler.step(n)

    # ----------------------------------------
    # test / inference
    # ----------------------------------------
    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.netG_forward()
        self.netG.train()

    # ----------------------------------------
    # get L, E, H image
    # ----------------------------------------
    def current_visuals(self, need_H=True):
        out_dict = OrderedDict()
        out_dict['L'] = self.L.detach()[0].float()
        out_dict['E'] = self.E.detach()[0].float()
        if need_H:
            out_dict['H'] = self.H.detach()[0].float()
        return out_dict

    #get learning rate
    def get_lr(self):
        for param_group in self.G_optimizer.param_groups:
            return param_group['lr']

def run(train_dir: str = "/p/scratch/deepacf/deeprain/bing/downscaling_maelstrom/train",
        val_dir: str = "/p/scratch/deepacf/deeprain/bing/downscaling_maelstrom/val",
        n_channels : int = 8, save_dir: str = "../results",
        checkpoint_save: int = 200,
        epochs: int = 2, type_net: str = "unet", patch_size: int = 2,
        window_size: int = 4, upscale_swinIR: int = 4, 
        upsampler_swinIR: str = "pixelshuffle", **kwargs):

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

    train_loader = create_loader(train_dir, patch_size=16)
    val_loader = create_loader(file_path=val_dir, mode="test", stat_path=train_dir, patch_size=16)
    print("The model {} is selected for training".format(type_net))
    if type_net == "unet":
        netG = unet(n_channels = n_channels)
        # Use input size
        flops, params = flopth(netG, in_size=((n_channels, 16, 16 ),))
        print("flops, params", flops, params)
    elif type_net == "swinIR":
        netG = swinIR(img_size=16, patch_size=patch_size, in_chans=n_channels,window_size=window_size,
                upscale=upscale_swinIR, upsampler=upsampler_swinIR)
           # Use input size
        #flops = netG.flops()
        flops, params = flopth(netG, in_size=((n_channels, 16, 16 ),))
        print("flops, params", flops,params)
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
                       # final_upsample="expand_first"

    elif type_net == "diffusion":
        conditional = kwargs["conditional"]
        timesteps = kwargs["timesteps"]
        netG = UNet_diff(img_size=160, n_channels=n_channels+1) #add one channel for the noise
        difussion = True

   
    else:
        raise NotImplementedError

    wandb.run.name = type_net
    netG_params = sum(p.numel() for p in netG.parameters() if p.requires_grad)
    print("Total trainable parameters:", netG_params)

    model = BuildModel(netG, save_dir = save_dir, difussion=difussion,
                       conditional=conditional, timesteps=timesteps)

    model.init_train()
    current_step = 0


    wandb.config = {
        "lr": model.G_optimizer_lr,
        "train_dir": train_dir,
        "val_dir": val_dir,
        "epochs": epochs,
        "window_size": window_size,
        "patch_size": patch_size
    }


    for epoch in range(epochs):
        for i, train_data in enumerate(train_loader):
            st = time.time()

            current_step += 1

            # -------------------------------
            # 1) update learning rate
            # -------------------------------
            model.update_learning_rate(current_step)
            lr = model.get_lr() #get learning rate

            # -------------------------------
            # 2) feed patch pairs
            # -------------------------------
            model.feed_data(train_data)
            #if i == 0:
            #    model.count_flops(train_data["H"])

            # -------------------------------
            # 3) optimize parameters
            # -------------------------------
            model.optimize_parameters()

            # -------------------------------
            # 6) Save model
            # -------------------------------
            if current_step == 1 or current_step % checkpoint_save == 0:
                model.save(current_step)
                print("Model Loss {} after step {}".format(model.G_loss, current_step))
                print("Model Saved")
                print("Time per step:", time.time() - st)
                with torch.no_grad():
                    val_loss = 0
                    for j, val_data in enumerate(val_loader):
                        if j < 5:
                            model.feed_data(val_data)
                            model.netG_forward()
                            val_loss = val_loss + model.G_lossfn(model.E, model.H)
                    val_loss = val_loss/5
                    print("training loss:", model.G_loss)
                    print("validation loss:", val_loss)
                wandb.log({"loss": model.G_loss, "val_loss": val_loss, "lr": lr})
                


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type = str, required = True,
                        help = "The directory where training data (.nc files) are stored")
    parser.add_argument("--val_dir", type = str, required = True,
                        help = "The directory where validation data (.nc files) are stored")
    parser.add_argument("--save_dir", type = str, help = "The checkpoint directory")
    parser.add_argument("--epochs", type = int, default = 2, help = "The checkpoint directory")
    parser.add_argument("--model_type", type = str, default = "unet", help = "The model type: unet, swinir")

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
        val_dir = args.val_dir,
        n_channels = 8,
        save_dir = args.save_dir,
        checkpoint_save = 1000,
        epochs = args.epochs,
        type_net = args.model_type,
        patch_size = args.patch_size,
        window_size = args.window_size,
        upscale_swinIR = args.upscale_swinIR,
        upsampler_swinIR = args.upsampler_swinIR,
        conditional=args.conditional,
        timesteps = args.timesteps)


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
