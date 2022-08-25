
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


PATH = "XXX"
test_dir = "/p/scratch/deepacf/deeprain/bing/downscaling_maelstrom/test"
test_loader = create_loader(test_dir)
type_net = "unet"
n_channels = 8

print("The model {} is selected for training".format(type_net))
if type_net == "unet":
    netG = unet(n_channels = n_channels)
elif type_net == "swinSR":
    netG = swinSR()
elif type_net == "vitSR":
    netG = vitSR(embed_dim = 768)
else:
    NotImplementedError

netG.load_state_dict(torch.load(PATH))
netG.eval()
idx = 0
for i, test_data in enumerate(test_loader):
    idx += 1
    netG.feed_data(test_data)
    netG.test()
    print("forecast loss ",  np.float(netG.G_loss))
    outputs = netG.current_visuals()
    print("maxium value of L image ", np.max(outputs["L"].numpy()))
    print("maxium value of E image ", np.max(outputs["E"].numpy()))