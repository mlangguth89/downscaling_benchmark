
# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)

# SPDX-License-Identifier: MIT

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2022-07-22"


import torch
from dataset_prep import PrecipDatasetInter
from torch.utils.data import DataLoader
from collections import OrderedDict
from torch.optim import lr_scheduler
from torch.optim import Adam
import sys
import os
import torch.nn as nn
sys.path.append('../')
from models.network_unet import UNet as net
import time



def create_loader(file_path: str = None, batch_size: int = 4, patch_size: int = 16,
                 vars_in: list = ["cape_in", "tclw_in", "sp_in", "tcwv_in", "lsp_in", "cp_in", "tisr_in",
                                  "yw_hourly_in"],
                 var_out: list = ["yw_hourly_tar"], sf: int = 10,
                 seed: int = 1234, loader_params: dict = None):

    """
    file_path : the path to the directory of .nc files
    vars_in   : the list contains the input variable namsaes
    var_out   : the list contains the output variable name
    batch_size: the number of samples per iteration
    patch_size: the patch size for low-resolution image,
                the corresponding high-resolution patch size should be muliply by scale factor (sf)
    sf        : the scaling factor from low-resolution to high-resolution
    seed      : specify a seed so that we can generate the same random index for shuffle function
    """

    dataset = PrecipDatasetInter(file_path, batch_size, patch_size, vars_in, var_out, sf, seed)

    return torch.utils.data.DataLoader(dataset, batch_size=None)



class BuildModel:
    def __init__(self, netG, G_lossfn_type: str = "l2", G_optimizer_type: str = "adam",
                 G_optimizer_lr: float = 0.0002, G_optimizer_betas: list = [0.9, 0.999],
                 G_optimizer_wd: int= 0, save_dir: str = "../results"):

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
                                                        0.5))

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
        print("datat[L] shape",data["L"].shape)
        self.L = data['L']
        if need_H:
            self.H = data['H']

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self):
        self.E = self.netG(self.L)[:,0,:,:]

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




train_file_path = "/p/scratch/deepacf/deeprain/bing/downscaling_maelstrom/train"
test_file_path = "/p/scratch/deepacf/deeprain/bing/downscaling_maelstrom/test"

train_loader = create_loader(train_file_path)
test_loader = create_loader(test_file_path)
epochs = 2
netG = net(n_channels=8)
netG_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total trainalbe parameters:",netG_params)
model = BuildModel(netG)
model.init_train()
current_step = 0
CHECKPOINT_SAVE = 200 #how many steps to save checkpoint

for epoch in range(epochs):
    for i, train_data in enumerate(train_loader):
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

        # -------------------------------
        # 6) Save model
        # -------------------------------
        if current_step == 1 or current_step % CHECKPOINT_SAVE == 0:
            model.save(current_step)
            print("Model Loss {} after step {}".format(model.G_loss, current_step))
            print("Model Saved")
            print("Time per step:", time.time() - st)







