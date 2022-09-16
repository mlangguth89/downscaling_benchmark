
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
from models.network_swinir import SwinIR as swinSR
#from models.network_vanilla_swin_transformer import SwinTransformerSR as swinSR
from models.network_vit import TransformerSR as vitSR
from utils.data_loader import create_loader
from main_scripts.main_train import BuildModel
import wandb
import os
import json
from datetime import datetime
import xarray as xr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type = str, required = True,
                        default = "/p/scratch/deepacf/deeprain/bing/downscaling_maelstrom/test",
                        help = "The directory where test data (.nc files) are stored")
    parser.add_argument("--save_dir", type = str, help = "The output directory")
    parser.add_argument("--model_type", type = str, default = "unet", help = "The model type: unet, swinir")
    parser.add_argument("--mode", type = str, default = "test", help = "The mode type: train, test")
    parser.add_argument("--k", type = int, default = 0.01, help = "The parameter for log-transform")
    parser.add_argument("--stat_dir", type = str, required = True,
                        default = "/p/scratch/deepacf/deeprain/bing/downscaling_maelstrom/train",
                        help = "The directory where the statistics json file of training data is stored")    
    parser.add_argument("--checkpoint_dir", type = str, required = True, help = "Please provide the checkpoint directory")

    args = parser.parse_args()

    print("The model {} is selected for training".format(args.model_type))
    if args.model_type == "unet":
        netG = unet(n_channels = 8)
    elif args.model_type == "swinSR":
        netG = swinSR(img_size=16,patch_size=1,in_chans=8,window_size=8,upscale=4,upsampler='nearest+conv')
    elif args.model_type == "vitSR":
        netG = vitSR(embed_dim = 768)
    else:
        NotImplementedError()

    model = BuildModel(netG)
    model.define_loss()
    total_loss = 0.
    test_len = []
    test_loader = create_loader(file_path=args.test_dir, mode=args.mode, stat_path=args.stat_dir)

    stat_file = os.path.join(args.stat_dir, "statistics.json")
    with open(stat_file,'r') as f:
        stat_data = json.load(f)
    vars_in_patches_mean = stat_data['yw_hourly_in_mean']
    vars_in_patches_std = stat_data['yw_hourly_in_std']
    vars_out_patches_mean = stat_data['yw_hourly_tar_mean']
    vars_out_patches_std = stat_data['yw_hourly_tar_std']

    with torch.no_grad():
        model.netG.load_state_dict(torch.load(args.checkpoint_dir))
        idx = 0
        input_list = []
        output_list = []
        pred_list = []
        ref_list = []
        cidx_list = []
        times_list = []

        for i, test_data in enumerate(test_loader):
            idx += 1
            cidx_temp = test_data["idx"].numpy()
            times_temp = test_data["T"].numpy()
            print('times: {}'.format(times_temp))
            print('cidx: {}'.format(cidx_temp))
            cidx_list.append(cidx_temp)
            times_list.append(times_temp)

            model.feed_data(test_data)
            model.netG_forward()

            # log-transform -> log(x+k)-log(k)
            input_vars = test_data["L"].numpy()
            print('input_vars shape: {}'.format(input_vars.shape))
            input_temp = np.squeeze(input_vars[:,-1,:,:])*vars_in_patches_std+vars_in_patches_mean
            #input_temp = np.exp(input_temp+np.log(args.k))-args.k
            input_list.append(input_temp)     
            output_temp = test_data["H"].numpy()*vars_out_patches_std+vars_out_patches_mean
            #output_temp = np.exp(output_temp+np.log(args.k))-args.k
            output_list.append(output_temp)

            pred_temp = model.E.numpy()*vars_out_patches_std+vars_out_patches_mean
            #pred_temp = np.exp(pred_temp+np.log(args.k))-args.k
            pred_list.append(pred_temp)
            ref_temp = model.H.numpy()*vars_out_patches_std+vars_out_patches_mean
            #ref_temp = np.exp(ref_temp+np.log(args.k))-args.k
            ref_list.append(ref_temp)            
            print('model.E shape: {}'.format(model.E.shape))
            #G_loss = model.G_lossfn(model.E, model.H)
            #print("forecast loss ", np.float(G_loss))
        
        cidx = np.squeeze(np.concatenate(cidx_list,0))
        times = np.concatenate(times_list,0)
        pred = np.concatenate(pred_list,0)
        ref = np.concatenate(ref_list,0)
        intL = np.concatenate(input_list,0)
        outH = np.concatenate(output_list,0)

        datetimes = []
        for i in range(times.shape[0]):
            times_str = str(times[i][0])+str(times[i][1]).zfill(2)+str(times[i][2]).zfill(2)+str(times[i][3]).zfill(2)
            print('times_str: {}'.format(times_str))
            datetimes.append(datetime.strptime(times_str,'%Y%m%d%H'))

        print('cidx shape:{}'.format(cidx.shape))
        ds = xr.Dataset(
            data_vars=dict(
                inputs=(["time", "lat_in", "lon_in"], intL),
                outputs=(["time", "lat", "lon"], outH),
                fcst=(["time", "lat", "lon"], pred),
                refe=(["time", "lat", "lon"], ref),
            ),
            coords=dict(
                time=datetimes,
                pitch_idx=cidx,
            ),
            attrs=dict(description="Precipitation downscaling data."),
            )

        os.makedirs(args.save_dir,exist_ok=True)
        ds.to_netcdf(os.path.join(args.save_dir,'prcp_downs_'+args.model_type+'.nc'))

if __name__ == '__main__':
    main()
