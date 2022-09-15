import argparse
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import json
from collections import OrderedDict
import os
import random
import numpy as np
import torch.utils.data as data
import math
import xarray as xr
from collections import OrderedDict
import time
import json
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim import Adam

import sys
sys.path.append('../')
from models.network_unet import UNet as unet
from utils.data_loader import create_loader
from main_scripts.dataset_temp_v2 import CustomTemperatureDataset

# parameters
CHECKPOINT_SAVE = 200 # how many steps to save checkpoint
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get data
fl = "/p/scratch/deepacf/maelstrom/maelstrom_data/ap5_michael/preprocessed_era5_crea6/netcdf_data/all_files/preproc_era5_crea6_train.nc" #C:\\Users\\max_b\\PycharmProjects\\downscaling_maelstrom\\preproc_era5_crea6_small.nc

dataset = CustomTemperatureDataset(file_path=fl)
train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

netG = unet(n_channels=9)
netG.to(device)
print()

class BuildModel:
    def __init__(self, netG, G_lossfn_type="l2", G_optimizer_type="adam",
                 G_optimizer_lr=0.0002, G_optimizer_betas=[0.9, 0.999], G_optimizer_wd=0, save_dir="../results"):
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

    def init_train(self):
        self.netG.train()
        self.define_loss()
        self.define_optimizer()

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

        self.G_optimizer = Adam(G_optim_params, lr = self.G_optimizer_lr,
                                betas = self.G_optimizer_betas,
                                weight_decay = self.G_optimizer_wd)

    # ----------------------------------------
    # define scheduler, only "MultiStepLR"
    # ----------------------------------------
    def define_scheduler(self):
        self.schedulers.append(lr_scheduler.MultiStepLR(self.G_optimizer,
                                                        [250000, 400000, 450000, 475000, 500000],
                                                        0.5
                                                        ))

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
        self.L = data[0].to(device)
        if need_H:
            self.H = data[1].to(device)

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self):
        self.E = self.netG(self.L).to(device)

    # ----------------------------------------
    # update parameters and get loss
    # ----------------------------------------
    def optimize_parameters(self, current_step):
        self.G_optimizer.zero_grad()
        self.netG_forward()
        self.G_loss = self.G_lossfn(self.E, self.H)
        self.G_loss.backward()

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = BuildModel(netG)

model.init_train()
current_step = 0


for epoch in range(30):  # keep running
    st_e = time.time()
    for i, train_data in enumerate(train_dataloader):
        st = time.time()

        current_step += 1

        # -------------------------------
        # 1) update learning rate
        # -------------------------------
        model.update_learning_rate(current_step)

        # -------------------------------
        # 2) feed patch pairs
        # -------------------------------
        model.feed_data(train_data)

        # -------------------------------
        # 3) optimize parameters
        # -------------------------------
        model.optimize_parameters(current_step)



        # if current_step == 1 or current_step % 100 == 0:
        #
        #     print("Model Loss {} after step {}".format(model.G_loss, current_step))
        #     print("Time per step:", time.time() - st)

    print("Model Loss {} after epoch {}".format(model.G_loss, epoch))
    print("Time per epoch:", time.time() - st_e)



print("training is done")
