# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT


__email__ = "b.gong@fz-juelich.de"
__author__ = "Maxim Bragilovski"
__date__ = "2022-09-02"

import sys

sys.path.append('../')
import xarray as xr
import time
import torch
import numpy as np
import pathlib
import math
import torchvision
import os
import json
from handle_data.handle_data_unet import HandleUnetData


class TempDatasetInter(torch.utils.data.IterableDataset):
    """
    This is the class used for generate dataset generator for precipitation downscaling
    """

    def __init__(self, file_path: str = None, batch_size: int = 4, verbose: int = 0, seed: int = 1234):
        """
        file_path : the path to the .nc dataset
        batch_size: the number of samples per iteration
        verbose: specify the kind of the dataset, 0 for train and 1 for validation
        seed      : specify a seed so that we can generate the same random index for shuffle function
        """

        super(TempDatasetInter).__init__()

        self.ds_tar = None
        self.ds_in = None
        self.file_path = file_path
        self.batch_size = batch_size
        self.verbose = verbose
        self.seed = seed
        self.ds = xr.open_dataset(file_path)
        start = time.time()
        self.ds.load()
        end = time.time()
        print(f'loading took {(end - start) / 60} minutes')
        self.times = np.transpose(np.stack(
            [self.ds["time"].dt.year, self.ds["time"].dt.month, self.ds["time"].dt.day, self.ds["time"].dt.hour]))
        self.process_era5_netcdf()

        print("The total number of samples are:", self.ds.sizes['time'])

        self.n_samples = self.ds.sizes['time']
        self.idx_perm = self.shuffle()

        self.log = self.ds.sizes['rlon']
        self.lat = self.ds.sizes['rlat']
        # self.save_stats()

    def process_era5_netcdf(self):
        """
        process netcdf files: normalization,
        """

        def reshape_ds(ds):
            da = ds.to_array(dim="variables")
            da = da.transpose(..., "variables")
            return da

        ds_train = self.ds.sel(time=slice("2006-01-01", "2010-12-30")) #
        print(ds_train.sizes)
        start = time.time()
        da_train = reshape_ds(ds_train)
        end = time.time()
        print(f'reshaping took {(end-start)/60} minutes')
        norm_dims = ["time", "rlat", "rlon"]

        if self.verbose == 0:
            start = time.time()
            da_norm, mu, std = HandleUnetData.z_norm_data(da_train, dims=norm_dims, return_stat=True)
            end = time.time()
            print(f'normalization took {(end - start) / 60} minutes')
            for save in [(mu, 'mu'), (std, 'std')]:
                self.save_stats(save[0], save[1])
        if self.verbose == 1:
            mu_train = self.load_stats('mu')
            std_train = self.load_stats('std')
            da_norm = HandleUnetData.z_norm_data(da_train, mu=mu_train, std=std_train)

        da_norm = da_norm.astype(np.float32)

        def gen(darr_in, darr_tar):
            ds_train_in = []
            ds_train_tar = []
            ntimes = len(darr_in["time"])
            for t in range(ntimes):
                ds_train_in.append(darr_in.isel({"time": t}).values)
                ds_train_tar.append(darr_tar.isel({"time": t}).values)
            return ds_train_in, ds_train_tar

        start = time.time()
        da_in, da_tar = split_in_tar(da_norm)
        end = time.time()
        print(f'splitting took {(end - start) / 60} minutes')
        start = time.time()
        self.ds_in, self.ds_tar = gen(da_in, da_tar)
        end = time.time()
        print(f'generation took {(end - start) / 60} minutes')

    def shuffle(self):
        """
        shuffle the index
        """
        print("Shuffling the index ....")
        multiformer_np_rng = np.random.default_rng(self.seed)
        idx_perm = multiformer_np_rng.permutation(self.n_samples)

        # restrict to multiples of batch size
        idx = int(math.floor(self.n_samples / self.batch_size)) * self.batch_size

        idx_perm = idx_perm[:idx]

        return idx_perm

    def save_stats(self, to_save, name):
        dict_to_save = to_save.to_dict()
        # json_object = json.dump(dict_to_save)
        path = self.file_path.split('\\')
        path = '\\'.join(e for e in path[0:len(path) - 1]) + '\\' + name + '.json'
        with open(path, 'w') as f:
            json.dump(dict_to_save, f)

    def load_stats(self, name):
        path = self.file_path.split('\\')
        path = '\\'.join(e for e in path[0:len(path) - 1]) + '\\' + name + '.json'
        with open(path) as json_file:
            data = json.load(json_file)

        for key in data.keys():
            print(key)
        ds = xr.DataArray.from_dict(data)
        return xr.DataArray.from_dict(data)

    def __iter__(self):

        iter_start, iter_end = 0, int(len(self.idx_perm) / self.batch_size)
        self.idx = 0

        for bidx in range(iter_start, iter_end):

            # initialise x, y for each batch
            # x  stores the low resolution images, y for high resolution,
            # t is the corresponding timestamps, cidx is the index
            x = torch.zeros(self.batch_size, 9, self.lat, self.log)
            y = torch.zeros(self.batch_size, 2, self.lat, self.log)
            t = torch.zeros(self.batch_size, 4, dtype=torch.int)
            cidx = torch.zeros(self.batch_size, 1, dtype=torch.int)  # store the index

            for jj in range(self.batch_size):
                cid = self.idx_perm[self.idx]

                x[jj] = torch.from_numpy(self.ds_in[cid]).permute(2, 0, 1)
                y[jj] = torch.from_numpy(self.ds_tar[cid]).permute(2, 0, 1)
                t[jj] = torch.from_numpy(self.times[cid])
                cidx[jj] = torch.tensor(cid, dtype=torch.int)

                self.idx += 1
            yield {'L': x, 'H': y, "idx": cidx, "T": t}


def run():
    data_loader = TempDatasetInter(
        file_path="/p/scratch/deepacf/maelstrom/maelstrom_data/ap5_michael/preprocessed_era5_crea6/netcdf_data/all_files/preproc_era5_crea6_train.nc")
    # /p/scratch/deepacf/maelstrom/maelstrom_data/ap5_michael/preprocessed_era5_crea6/netcdf_data/preproc_era5_crea6_trainall_files/preproc_era5_crea6_train.nc # file_path="C:\\Users\\max_b\\PycharmProjects\\downscaling_maelstrom\\preproc_era5_crea6_train.nc")
    # C:\\Users\\max_b\\PycharmProjects\\downscaling_maelstrom\\preproc_era5_crea6_small.nc"
    print("created data_loader")
    for batch_idx, train_data in enumerate(data_loader):
        start = time.time()
        inputs = train_data["L"]
        target = train_data["H"]
        idx = train_data["idx"]
        times = train_data["T"]
        end = time.time()
        print(f'getting 1 batch took {(end - start) / 60} minutes')
        print("inputs", inputs.size())
        print("target", target.size())
        print("idx", idx)
        print("batch_idx", batch_idx)
        print("timestamps,", times)


def split_in_tar(da: xr.DataArray, target_var: str = "t2m") -> (xr.DataArray, xr.DataArray):
    """
    Split data array with variables-dimension into input and target data for downscaling.
    :param da: The unsplitted data array.
    :param target_var: Name of target variable which should consttute the first channel
    :return: The splitted data array.
    """
    invars = [var for var in da["variables"].values if var.endswith("_in")]
    tarvars = [var for var in da["variables"].values if var.endswith("_tar")]

    # ensure that ds_tar has a channel coordinate even in case of single target variable
    roll = False
    if len(tarvars) == 1:
        sl_tarvars = tarvars
    else:
        sl_tarvars = slice(*tarvars)
        if tarvars[0] != target_var:  # ensure that target variable appears as first channel
            roll = True

    da_in, da_tar = da.sel({"variables": invars}), da.sel(variables=sl_tarvars)
    if roll: da_tar = da_tar.roll(variables=1, roll_coords=True)

    return da_in, da_tar


if __name__ == "__main__":
    run()
