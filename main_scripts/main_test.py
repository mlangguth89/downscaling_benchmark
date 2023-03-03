
# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)

# SPDX-License-Identifier: MIT

__email__ = "b.gong@fz-juelich.de"
__author__ = "Bing Gong"
__date__ = "2022-08-22"

import argparse
import sys
import torch
import numpy as np
sys.path.append('../')
from models.network_unet import UNet as unet
from models.network_swinir import SwinIR as swinIR
from models.network_vit import TransformerSR as vitSR
from models.network_swinunet_sys import SwinTransformerSys as swinUnet
from models.network_diffusion import UNet_diff
from models.diffusion_utils import GaussianDiffusion
from utils.data_loader import create_loader
from main_scripts.main_train import BuildModel
#System packages
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

    n_channels = 10
    print("The model {} is selected for training".format(args.model_type))
    if args.model_type == "unet":
        netG = unet(n_channels = n_channels)
    elif args.model_type == "vitSR":
        netG = vitSR(embed_dim = 768)
    elif args.model_type == "swinUnet":
        netG = swinUnet(img_size=160, patch_size=2,
                        in_chans=n_channels,
                        num_classes=1,
                        embed_dim=96,
                        depths=[2, 2, 2],
                        depths_decoder=[2,2,2],
                        num_heads=[6,6,6],
                        window_size=4,
                        mlp_ratio=4,
                        qkv_bias=True,
                        qk_scale=None,
                        drop_rate=0.,
                        attn_drop_rate=0.,
                        drop_path_rate=0.1,
                        ape=False,
                        final_upsample="expand_first")
    elif args.model_type == "swinIR":
        netG = swinIR(img_size=16,
                      patch_size=4,
                      in_chans=n_channels,
                      window_size=2,
                      upscale= 4,
                      upsampler= "pixelshuffle") 
    elif args.model_type == "diffusion":
        netG = UNet_diff(n_channels = n_channels+1,
                         img_size=160)
        difussion = True
        gf = GaussianDiffusion(conditional=True,
                               schedule_opt="linear",
                               timesteps=200,
                               model=netG)
    else:
        NotImplementedError()

    if args.model_type == "diffusion":
        model = BuildModel(netG,  difussion=difussion,
                       conditional=True, timesteps=args.timesteps)

    else:
        model = BuildModel(netG)


    test_loader = create_loader(file_path=args.test_dir,
                                mode=args.mode,
                                stat_path=args.stat_dir)
    stat_file = os.path.join(args.stat_dir, "statistics.json")
    
    with open(stat_file,'r') as f:
        stat_data = json.load(f)
    vars_in_patches_mean  = stat_data['yw_hourly_in_mean']
    vars_in_patches_std   = stat_data['yw_hourly_in_std']
    vars_out_patches_mean = stat_data['yw_hourly_tar_mean']
    vars_out_patches_std  = stat_data['yw_hourly_tar_std']

    with torch.no_grad():
        model.netG.load_state_dict(torch.load(args.checkpoint_dir))
        idx = 0
        input_list = []
        pred_list = []
        ref_list = []
        cidx_list = []
        times_list = []
        noise_pred_list = []
        all_sample_list = [] #this is ony for difussion model inference
        for i, test_data in enumerate(test_loader):
            idx += 1
            batch_size = test_data["L"].shape[0]
            cidx_temp = test_data["idx"]
            times_temp = test_data["T"]
            cidx_list.append(cidx_temp.cpu().numpy())
            times_list.append(times_temp.cpu().numpy())
            model.feed_data(test_data)
            #we must place calculate the shape of input here, due to for diffussion model, 
            #The L is upsampling to higher resolution before feed into the model through 'feed_data' function
            image_size = model.L.shape[2]
            model.netG_forward()

            #Get the low resolution inputs
            input_vars = test_data["L"]
            input_temp = np.squeeze(input_vars[:,-1,:,:])*vars_in_patches_std+vars_in_patches_mean
            input_temp = np.exp(input_temp.cpu().numpy()+np.log(args.k))-args.k
            input_list.append(input_temp)
            if args.model_type == "diffusion":
                #now, we only use the unconditional difussion model, meaning the inputs are only noise.
                #This is the first test, later, we will figure out how to use conditioanl difussion model.
                print("Start reverse process")
                samples = gf.sample(image_size=image_size, batch_size=batch_size, channels=8, x_in=model.L)
                #chose the last channle and last varialbe (precipitation)
                sample_last = samples[-1] *vars_out_patches_std+vars_out_patches_mean
                # we can make some plot here
                #all_sample_list = all_sample_list.append(sample_last)
                pred_temp = sample_last * vars_out_patches_std + vars_out_patches_mean
                #pred_temp = np.exp(pred_temp.cpu().numpy()+np.log(args.k))-args.k
                ref_temp = model.H #this is the true noise
                noise_pred = model.E #predict the noise
                noise_pred_list.append(noise_pred.cpu().numpy())
            else:
                #Get the prediction values
                pred_temp = model.E.cpu().numpy() * vars_out_patches_std + vars_out_patches_mean
                pred_temp = np.exp(pred_temp+np.log(args.k))-args.k
                #Get the groud truth values
                ref_temp = model.H.cpu().numpy()*vars_out_patches_std+vars_out_patches_mean
                ref_temp = np.exp(ref_temp+np.log(args.k))-args.k

            ref_list.append(ref_temp)
            pred_list.append(pred_temp)   
        
        cidx = np.squeeze(np.concatenate(cidx_list,0))
        times = np.concatenate(times_list,0)
        pred = np.concatenate(pred_list,0)
        ref = np.concatenate(ref_list,0)
        intL = np.concatenate(input_list,0)
        datetimes = []

        for i in range(times.shape[0]):
            times_str = str(times[i][0])+str(times[i][1]).zfill(2)+str(times[i][2]).zfill(2)+str(times[i][3]).zfill(2)
            datetimes.append(datetime.strptime(times_str,'%Y%m%d%H'))

        if len(pred.shape) == 4:
            pred = pred[:, 0 , : ,:]
        if len(ref.shape) == 4:
            ref = ref[:, 0,: ,:]
        if len(intL.shape) == 4:
            intL = intL[:, 0,: ,:]

        if args.model_type == "diffusion":
            noiseP = np.concatenate(noise_pred_list,0)
            if len(noiseP.shape) == 4:
                noiseP = noiseP[:, 0, :, :]
            ds = xr.Dataset(
                data_vars = dict(
                    inputs = (["time", "lat_in", "lon_in"], intL),
                    fcst = (["time", "lat", "lon"], np.squeeze(pred)),
                    refe = (["time", "lat", "lon"], ref),
                    noiseP = (["time", "lat", "lon"], noiseP)
                ),
                coords = dict(
                    time = datetimes,
                    pitch_idx = cidx,
                    ),
                attrs = dict(description = "Precipitation downscaling data."),
                )
        else:
            ds = xr.Dataset(
                data_vars = dict(
                    inputs = (["time", "lat_in", "lon_in"], intL),
                    fcst = (["time", "lat", "lon"], np.squeeze(pred)),
                    refe = (["time", "lat", "lon"], ref),
                ),
                coords = dict(
                    time = datetimes,
                    pitch_idx = cidx,
                    ),
                attrs = dict(description = "Precipitation downscaling data."),
                )



        os.makedirs(args.save_dir,exist_ok=True)
        ds.to_netcdf(os.path.join(args.save_dir,'prcp_downs_'+args.model_type+'.nc'))

if __name__ == '__main__':
    cuda = torch.cuda.is_available()
    if cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    main()
