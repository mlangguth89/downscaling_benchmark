
# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)

# SPDX-License-Identifier: MIT

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2022-08-22"

import time
import argparse
import sys
import torch
import numpy as np
from dataset_prep import PrecipDatasetInter
sys.path.append('../')
from models.network_unet import UNet as unet
from models.network_vanilla_swin_transformerimport import SwinTransformerSR as swinSR
from models.network_vit import TransformerSR as vitSR
from utils.data_loader import create_loader
from main_scripts.main_train import BuildModel
import wandb






def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type = str, required = True,
                        default = "/p/scratch/deepacf/deeprain/bing/downscaling_maelstrom/test",
                        help = "The directory where test data (.nc files) are stored")
    parser.add_argument("--save_dir", type = str, help = "The output directory")
    parser.add_argument("--model_type", type = str, default = "unet", help = "The model type: unet, swinir")
    parser.add_argument("--checkpoint_dir", type = str, required = True, help = "Please provide the checkpoint directory")

    args = parser.parse_args()

    print("The model {} is selected for training".format(type_net))
    if args.model_type == "unet":
        netG = unet(n_channels = 8)
    elif args.model_type == "swinSR":
        netG = swinSR()
    elif args.model_type == "vitSR":
        netG = vitSR(embed_dim = 768)
    else:
        NotImplementedError()

    model = BuildModel(netG)
    model.test()
    total_loss = 0.
    test_len = []
    test_loader = create_loader(args.test_dor)

    with torch.no_grad():
        model.netG.load_state_dict(torch.load(args.test_dir))
        idx = 0
        for i, test_data in enumerate(test_loader):
            idx += 1
            model.feed_data(test_data)
            model.netG_forward()
            G_loss = model.G_lossfn(model.E, model.H)
            print("forecast loss ", np.float(G_loss))



if __name__ == '__main__':
    main()
