
import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
import time
import json
import sys

sys.path.append('../')
from handle_data.handle_data_unet import HandleUnetData
from torch.utils.data import DataLoader

class CustomTemperatureDataset(Dataset):
    def __init__(self, file_path: str = None, batch_size: int = 4, verbose: int = 0, seed: int = 1234):
        self.ds_tar = None
        self.ds_in = None
        self.file_path = file_path
        self.verbose = verbose
        self.ds = xr.open_dataset(file_path)
        self.log = self.ds.sizes['rlon']
        self.lat = self.ds.sizes['rlat']
        self.times = np.transpose(np.stack(
            [self.ds["time"].dt.year, self.ds["time"].dt.month, self.ds["time"].dt.day, self.ds["time"].dt.hour]))

        self.process_era5_netcdf()


    def process_era5_netcdf(self):
        """
        process netcdf files: normalization,
        """

        def reshape_ds(ds):
            da = ds.to_array(dim="variables")
            da = da.transpose(..., "variables")
            return da

        # ds_train = self.ds.sel(time=slice("2006-01-01", "2009-01-01"))

        start = time.time()
        da_train = reshape_ds(self.ds)
        end = time.time()
        print(f'Reshaping took {(end - start) / 60} minutes')

        self.n_samples = da_train.sizes['time']
        print(da_train.sizes)

        norm_dims = ["time", "rlat", "rlon"]
        if self.verbose == 0:
            start = time.time()
            da_norm, mu, std = HandleUnetData.z_norm_data(da_train, dims=norm_dims, return_stat=True)
            end = time.time()
            print(f'Normalization took {(end - start) / 60} minutes')
            for save in [(mu, 'mu'), (std, 'std')]:
                self.save_stats(save[0], save[1])
        if self.verbose == 1:
            mu_train = self.load_stats('mu')
            std_train = self.load_stats('std')
            da_norm = HandleUnetData.z_norm_data(da_train, mu=mu_train, std=std_train)

        da_norm = da_norm.astype(np.float32)
        start = time.time()
        da_norm.load()
        end = time.time()
        print(f'Loading took {(end - start) / 60} minutes')
        def gen(darr_in, darr_tar):
            ds_train_in = []
            ds_train_tar = []
            ntimes = len(darr_in["time"])
            for t in range(ntimes):
                ds_train_in.append(torch.from_numpy(darr_in.isel({"time": t}).values).permute(2, 0, 1))
                vector_tar = torch.from_numpy(darr_tar.isel({"time": t}).values[:, :, 1][..., np.newaxis])
                ds_train_tar.append(vector_tar.permute(2, 0, 1))   # [:, :, 1]
            return torch.stack(ds_train_in), torch.stack(ds_train_tar)

        start = time.time()
        da_in, da_tar = split_in_tar(da_norm)
        end = time.time()
        print(f'splitting took {(end - start) / 60} minutes')
        start = time.time()
        self.ds_in, self.ds_tar = gen(da_in, da_tar)

        end = time.time()
        print(f'generation took {(end - start) / 60} minutes')


    def save_stats(self, to_save, name):
        """
        Saving the statistics of the train data set to a json file
        """
        dict_to_save = to_save.to_dict()
        # json_object = json.dump(dict_to_save)
        path = self.file_path.split('\\')
        path = '\\'.join(e for e in path[0:len(path) - 1]) + '\\' + name + '.json'
        with open(path, 'w') as f:
            json.dump(dict_to_save, f)

    def load_stats(self, name):
        """
        Loading the statistics of the train data of normalization a validation/test dataset
        """
        path = self.file_path.split('\\')
        path = '\\'.join(e for e in path[0:len(path) - 1]) + '\\' + name + '.json'
        with open(path) as json_file:
            data = json.load(json_file)

        for key in data.keys():
            print(key)

        return xr.DataArray.from_dict(data)

    def __len__(self):
        return len(self.ds_in)

    def __getitem__(self, idx):
        return self.ds_in[idx], self.ds_tar[idx]

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

def run():
    data_loader = CustomTemperatureDataset(
        file_path="C:\\Users\\max_b\\PycharmProjects\\downscaling_maelstrom\\preproc_era5_crea6_small.nc")
    #   /p/scratch/deepacf/maelstrom/maelstrom_data/ap5_michael/preprocessed_era5_crea6/netcdf_data/all_files/preproc_era5_crea6_train.nc
    # C:\\Users\\max_b\\PycharmProjects\\downscaling_maelstrom\\preproc_era5_crea6_small.nc"
    train_dataloader = DataLoader(data_loader, batch_size=64, shuffle=True)
    train_features, train_labels = next(iter(train_dataloader))
    print(train_features, train_labels)
    # print("created data_loader")
    # for batch_idx, train_data in enumerate(data_loader):
    #     start = time.time()
    #     inputs = train_data["L"]
    #     target = train_data["H"]
    #     idx = train_data["idx"]
    #     times = train_data["T"]
    #     end = time.time()
    #     print(f'getting 1 batch took {(end - start)} seconds')
    #     print("inputs", inputs.size())
    #     print("target", target.size())
    #     print("idx", idx)
    #     print("batch_idx", batch_idx)
    #     print("timestamps,", times)

if __name__ == "__main__":
    run()