
__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2022-12-08"

import sys
sys.path.append('../')
from models.diffusion_utils import GaussianDiffusion
from models.network_unet import Upsampling
from utils.data_loader import create_loader

import time
from collections import OrderedDict
from torch.optim import Adam
import torch
import torch.nn as nn
import os
from torch.optim import lr_scheduler
import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class BuildModel:
    def __init__(self, netG,
                 G_lossfn_type: str = "l1",
                 G_optimizer_type: str = "adam",
                 G_optimizer_lr: float = 5.e-02,
                 G_optimizer_betas: list = [0.9, 0.999],  #5.e-05
                 G_optimizer_wd: int = 0,
                 save_dir: str = "../results",
                 train_loader: object = None,
                 val_loader: object = None,
                 epochs: int = 70,
                 decay_start: int = 5,
                 decay_end: int = 30,
                 dataset_type: str = 'precipitation',
                 diffusion=False,
                 save_freq=1000,
                 **kwargs):

        """

        :param netG:
        :param G_lossfn_type:
        :param G_optimizer_type:
        :param G_optimizer_lr:
        :param save_dir: str, the save model path
        :param diffusion: if enable diffusion, the conditional must be defined
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
        self.diffusion = diffusion
        self.save_dir = save_dir
        self.dataset_type=dataset_type

        self.epochs = epochs
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.decay_start = decay_start
        self.decay_end = decay_end
        self.save_freq = save_freq


        if diffusion:
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
        self.L = data['L'].to(device)
        if self.diffusion:
            upsampling = Upsampling(in_channels = 8)
            self.L = upsampling(self.L)
        self.H = data['H'].to(device)

    def count_flops(self,data):
        # Count the number of FLOPs
        c_ops = count_ops(self.netG,data)
        print("The number of FLOPS is:",c_ops )

    # ----------------------------------------
    # feed L to netG
    # ----------------------------------------
    def netG_forward(self):

        if not self.diffusion:
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


    #train model
    def fit(self):
        self.init_train()
        current_step = 0
        for epoch in range(self.epochs):
            for i, train_data in enumerate(self.train_loader):
                st = time.time()

                current_step += 1

                # -------------------------------
                # 1) update learning rate
                # -------------------------------
                # if epoch > self.decay_start and epoch < self.decay_start:
                # self.update_learning_rate(current_step)

                lr = self.get_lr()  # get learning rate

                # -------------------------------
                # 2) feed patch pairs
                # -------------------------------
                self.feed_data(train_data)

                # -------------------------------
                # 3) optimize parameters
                # -------------------------------
                self.optimize_parameters()

                self.schedulers[0].step(current_step)

                # -------------------------------
                # 6) Save model
                # -------------------------------
                if current_step % self.save_freq == 0 or current_step == 1:
                    self.save(current_step)
            print("Model Loss {} after step {}".format(self.G_loss, current_step))
            print("Model Saved")
            print("Time per step:", time.time() - st)
            wandb.log({"loss": self.G_loss, "lr": lr})

            with torch.no_grad():
                val_loss = 0
                counter = 0
                for j, val_data in enumerate(self.val_loader):
                    counter = counter + 1
                    self.feed_data(val_data)
                    self.netG_forward()
                    val_loss = val_loss + self.G_lossfn(self.E, self.H).detach()
                val_loss = val_loss / counter
                print("training loss:", self.G_loss.item())
                print("validation loss:", val_loss.item())
                print("lr", lr)






