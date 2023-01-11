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
import torch.nn as nn

sys.path.append('../')
from models.network_unet import UNet as unet
from models.network_unet_temp import UNet as unet_temp
from models.network_vanilla_swin_transformer import SwinTransformerSR as swinSR
from models.network_vit import TransformerSR as vitSR
from models.network_critic import Discriminator as critic
from utils.data_loader import create_loader
from train_scripts.wgan_train import BuildWGANModel
import wandb

os.environ["WANDB_MODE"] = "offline"
# os.environ["WANDB_API_KEY"] = key
wandb.init(project="Precip_downscaling", reinit=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BuildModel:
    def __init__(self, netG, G_lossfn_type: str = "l2", G_optimizer_type: str = "adam",
                 G_optimizer_lr: float = 0.0002, G_optimizer_betas: list = [0.9, 0.999],
                 G_optimizer_wd: int = 0, save_dir: str = "../results",
                 train_loader: object = None, epochs: int = 30, checkpoint_save: int = 200):

        # ------------------------------------
        # define network
        # ------------------------------------
        self.netG = netG
        self.G_lossfn_type = G_lossfn_type
        self.G_optimizer_type = G_optimizer_type
        self.G_optimizer_lr = G_optimizer_lr
        self.G_optimizer_betas = G_optimizer_betas
        self.G_optimizer_wd = G_optimizer_wd
        self.schedulers = []
        self.save_dir = save_dir

        self.checkpoint_save = checkpoint_save
        self.epochs = epochs
        self.train_loader = train_loader
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

        self.G_optimizer = Adam(G_optim_params, lr=self.G_optimizer_lr,
                                betas=self.G_optimizer_betas,
                                weight_decay=self.G_optimizer_wd)

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                        milestones=[1, 2, 3],
                                                        gamma=0.1))

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
    def feed_data(self, data, need_H=True):
        # print("datat[L] shape", data["L"].shape)
        self.L = data[0].to(device)
        if need_H:
            self.H = data[1].to(device)

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self):
        self.E = self.netG(self.L)[:, 0, :, :]

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
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

    # get learning rate
    def get_lr(self):
        for param_group in self.G_optimizer.param_groups:
            return param_group['lr']

    def fit(self):
        checkpoint_save = 10000
        self.init_train()
        current_step = 0

        for epoch in range(self.epochs):
            for i, train_data in enumerate(self.train_loader):
                st = time.time()

                current_step += 1

                # -------------------------------
                # 1) update learning rate
                # -------------------------------
                self.update_learning_rate(current_step)
                lr = self.get_lr()  # get learning rate

                # -------------------------------
                # 2) feed patch pairs
                # -------------------------------
                self.feed_data(train_data)

                # -------------------------------
                # 3) optimize parameters
                # -------------------------------
                self.optimize_parameters(current_step)

                # -------------------------------
                # 6) Save model
                # -------------------------------
            # if current_step == 1 or current_step % checkpoint_save == 0:
            self.save(current_step)
            print("Model Loss {} after step {}".format(self.G_loss, epoch))
            print("Model Saved")
            print("Time per step:", time.time() - st)
            wandb.log({"loss": self.G_loss, "lr": lr})


def run(train_dir: str = "/p/scratch/deepacf/deeprain/bing/downscaling_maelstrom/train", test_dir: str = None,
        n_channels: int = 8, save_dir: str = "../results", checkpoint_save: int = 200,
        epochs: int = 2, type_net: str = "unet", batch_size: int = 4, dataset_type: str = "temperature",
        args: dict = None):
    """
    :param train_dir       : the directory that contains the training dataset NetCDF files
    :param test_dir        : the directory that contains the testing dataset NetCDF files
    :param checkpoint_save : how many steps to save checkpoint
    :param n_channels      : the number of input variables/channels
    :param save_dir        : the directory where the checkpoint results are save
    :param epochs          : the number of epochs
    :param batch_size      : the batch size
    :param type_net        : the type of the models
    :param dataset_type    : the type of the dataset
    :param args            : the args of the program
    """

    train_loader = create_loader(train_dir, batch_size=batch_size, dataset_type=dataset_type)
    test_loader = create_loader(test_dir, batch_size=batch_size, dataset_type=dataset_type, verbose=1)

    print("The model {} is selected for training".format(type_net))
    if type_net == "unet":
        netG = unet(n_channels=n_channels)
    elif type_net == "swinSR":
        netG = swinSR()
    elif type_net == "vitSR":
        netG = vitSR(embed_dim=768)
    elif type_net == "wgan":
        netG = unet_temp(n_channels=n_channels)
        netC = critic((1, 160, 160))
    else:
        NotImplementedError

    netG_params = sum(p.numel() for p in netG.parameters() if p.requires_grad)
    if type_net == "wgan":
        netC_params = sum(p.numel() for p in netC.parameters() if p.requires_grad)
        print("Total trainalbe parameters of the generator:", netG_params)
        print("Total trainalbe parameters of the critic:", netC_params)
    else:
        print("Total trainalbe parameters:", netG_params)

    if type_net == "wgan":
        model = BuildWGANModel(generator=netG, save_dir=save_dir, critic=netC, train_dataloader=train_loader,
                      val_dataloader=test_loader, checkpoint_save=checkpoint_save, hparams=args)
    else:
        model = BuildModel(netG, save_dir=save_dir, train_loader=train_loader, epochs=epochs,
                       checkpoint_save=checkpoint_save)
    model.fit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=False,
                        default="C:\\Users\\max_b\\PycharmProjects\\downscaling_maelstrom\\preproc_era5_crea6_small.nc",
                        help="The directory where training data (.nc files) are stored")
    parser.add_argument("--test_dir", type=str, required=False,
                        default="C:\\Users\\max_b\\PycharmProjects\\downscaling_maelstrom\\preproc_era5_crea6_small.nc",
                        help="The directory where testing data (.nc files) are stored")
    parser.add_argument("--save_dir", type=str,
                        default="C:\\Users\\max_b\\PycharmProjects\\downscaling_maelstrom\\saves",
                        help="The checkpoint directory")
    parser.add_argument("--epochs", type=int, default=2, help="The checkpoint directory")
    parser.add_argument("--model_type", type=str, default="unet", help="The model type: unet, swinir, wgan")
    parser.add_argument("--dataset_type", type=str, default="temperature",
                        help="The dataset type: temperature, precipitation")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--critic_iterations", type=float, default=4, help="The checkpoint directory")
    parser.add_argument("--lr_gn", type=float, default=5.e-05, help="The checkpoint directory")
    parser.add_argument("--lr_gn_end", type=float, default=5.e-06, help="The checkpoint directory")
    parser.add_argument("--lr_critic", type=float, default=1.e-06, help="The checkpoint directory")
    parser.add_argument("--decay_start", type=int, default=25, help="The checkpoint directory")
    parser.add_argument("--decay_end", type=int, default=50, help="The checkpoint directory")
    parser.add_argument("--lambada_gp", type=float, default=10, help="The checkpoint directory")
    parser.add_argument("--recon_weight", type=float, default=1000, help="The checkpoint directory")

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # save the args to the checkpoint directory
    with open(os.path.join(args.save_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(args), sort_keys=True, indent=4))

    run(train_dir=args.train_dir,
        test_dir=args.test_dir,
        n_channels=9,
        save_dir=args.save_dir,
        checkpoint_save=10000,
        epochs=args.epochs,
        type_net=args.model_type,
        batch_size=args.batch_size,
        dataset_type=args.dataset_type,
        args=args)


if __name__ == '__main__':
    main()
