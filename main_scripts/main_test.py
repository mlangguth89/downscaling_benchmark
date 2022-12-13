
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
sys.path.append('../')
from models.network_unet import UNet as unet
from models.network_swinir import SwinIR as swinSR
#from models.network_vanilla_swin_transformer import SwinTransformerSR as swinSR
from models.network_vit import TransformerSR as vitSR
from models.network_swinunet_sys import SwinTransformerSys as swinUnet
from models.diffusion_utilse import sample
from models.network_diffusion  import UNet_diff
from models.diffusion_utilise import GaussianDiffusion
from utils.data_loader import create_loader
from main_scripts.main_train import BuildModel

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

    # PARAMETERS FOR SWIN-IR & SWIN-UNET
    parser.add_argument("--patch_size", type = int, default = 2)
    parser.add_argument("--window_size", type = int, default = 4)

    # PARAMETERS FOR SWIN-IR
    parser.add_argument("--upscale_swinIR", type = int, default = 4)
    parser.add_argument("--upsampler_swinIR", type = str, default = "pixelshuffle")

    args = parser.parse_args()

    n_channels = 8
    print("The model {} is selected for training".format(args.model_type))
    if args.model_type == "unet":
        netG = unet(n_channels = n_channels)
    elif args.model_type == "swinSR":
        netG = swinSR(img_size=16,patch_size=args.patch_size,in_chans=n_channels,window_size=args.window_size,
                upscale=args.upscale_swinIR,upsampler=args.upsampler_swinIR)
        # netG = swinSR(img_size=16,patch_size=1,in_chans=8,window_size=8,upscale=4,upsampler='nearest+conv')
    elif args.model_type == "vitSR":
        netG = vitSR(embed_dim = 768)
    elif args.model_type == "swinUnet":
        netG = swinUnet(img_size=160,patch_size=args.patch_size,
                        in_chans=n_channels,num_classes=1,embed_dim=96,
                        depths=[2,2,2],depths_decoder=[2,2,2],num_heads=[6,6,6],
                        window_size=args.window_size,mlp_ratio=4,qkv_bias=True,qk_scale=None,
                        drop_rate=0.,attn_drop_rate=0.,drop_path_rate=0.1,ape=False,final_upsample="expand_first")

    elif args.model_type == "difussion":
        netG = UNet_diff(n_channels = n_channels)
        difussion = True
        gf = GaussianDiffusion(conditional=True, schedule_opt="linear", timesteps=200, model=netG)
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
        all_sample_list = [] #this is ony for difussion model inference
        for i, test_data in enumerate(test_loader):
            idx += 1
            batch_size = test_data.shape[0]
            image_size = test_data.shape[1]
            cidx_temp = test_data["idx"].numpy()
            times_temp = test_data["T"].numpy()
            #print('times: {}'.format(times_temp))
            #print('cidx: {}'.format(cidx_temp))
            cidx_list.append(cidx_temp)
            times_list.append(times_temp)

            model.feed_data(test_data)
            model.netG_forward()


            # log-transform -> log(x+k)-log(k)
            input_vars = test_data["L"].numpy()
            #print('input_vars shape: {}'.format(input_vars.shape))
            input_temp = np.squeeze(input_vars[:,-1,:,:])*vars_in_patches_std+vars_in_patches_mean
            #input_temp = np.exp(input_temp+np.log(args.k))-args.k
            input_list.append(input_temp)

            if args.model_type == "diffusion":
                #now, we only use the unconditional difussion model, meaning the inputs are only noise.
                #This is the first test, later, we will figure out how to use conditioanl difussion model.
                samples = gf.sample(image_size = image_size, batch_size = batch_size, channels = 8 )
                #chose the last channle and last varialbe (precipitation)
                sample_last = samples[-1][-1].numpy()*vars_out_patches_std+vars_out_patches_mean

                # we can make some plot here
                all_sample_list = all_sample_list.append(sample_last)

            else:
                output_temp = test_data["H"].numpy()*vars_out_patches_std+vars_out_patches_mean
            #output_temp = np.exp(output_temp+np.log(args.k))-args.k
            output_list.append(output_temp)

            pred_temp = model.E.numpy()*vars_out_patches_std+vars_out_patches_mean
            #pred_temp = np.exp(pred_temp+np.log(args.k))-args.k
            pred_list.append(pred_temp)
            ref_temp = model.H.numpy()*vars_out_patches_std+vars_out_patches_mean
            #ref_temp = np.exp(ref_temp+np.log(args.k))-args.k
            ref_list.append(ref_temp)            
            #print('model.E shape: {}'.format(model.E.shape))
            #G_loss = model.G_lossfn(model.E, model.H)
            #print("forecast loss ", np.float(G_loss))
        
        cidx = np.squeeze(np.concatenate(cidx_list,0))
        times = np.concatenate(times_list,0)
        pred = np.concatenate(pred_list,0)
        ref = np.concatenate(ref_list,0)
        intL = np.concatenate(input_list,0)
        outH = np.concatenate(output_list,0)

        print('pred shape: {}'.format(pred.shape))
        print('ref shape: {}'.format(ref.shape))
        print('intL shape: {}'.format(intL.shape))
        print('outH shape: {}'.format(outH.shape))

        datetimes = []
        for i in range(times.shape[0]):
            times_str = str(times[i][0])+str(times[i][1]).zfill(2)+str(times[i][2]).zfill(2)+str(times[i][3]).zfill(2)
            #print('times_str: {}'.format(times_str))
            datetimes.append(datetime.strptime(times_str,'%Y%m%d%H'))

        #print('cidx shape:{}'.format(cidx.shape))
        ds = xr.Dataset(
            data_vars=dict(
                inputs=(["time", "lat_in", "lon_in"], intL),
                outputs=(["time", "lat", "lon"], outH),
                fcst=(["time", "lat", "lon"], np.squeeze(pred)),
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
