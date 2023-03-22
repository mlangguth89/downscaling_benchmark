import os, sys, glob
sys.path.append('../')
import torch
from torch.utils.data import Dataset
import xarray as xr
import logging
from handle_data.handle_data_class import HandleDataClass
from handle_data.all_normalizations import ZScore


class test_dataset(Dataset):
    def __init__(self, fdata_test: str = None, norm_dir: str = None, ds_dict: dict = None, logger: logging = None):
        """

        """
        ds_test = xr.open_dataset(fdata_test)
        # ds_test = ds_test.sel(time=slice("2006-01-01", "2006-01-05"))

        self.da_test = HandleDataClass.reshape_ds(ds_test.astype("float32", copy=False))

        # perform normalization
        js_norm = os.path.join(norm_dir, "norm.json")
        logger.debug("Read normalization file for subsequent data transformation.")
        self.norm = ZScore(ds_dict["norm_dims"])
        self.norm.read_norm_from_file(js_norm)
        self.da_test = self.norm.normalize(self.da_test)

        self.da_test_in, self.da_test_tar = HandleDataClass.split_in_tar(self.da_test)
        self.tar_varname = self.da_test_tar['variables'].values[0]
        self.ground_truth = ds_test[self.tar_varname].astype("float32", copy=False)
        logger.info(f"Variable {self.tar_varname} serves as ground truth data.")
        self.da_test_in_tensor, self.da_test_tar_tensor = HandleDataClass.gen(self.da_test_in, self.da_test_tar)

        logger.info(f"Variable {self.tar_varname} serves as ground truth data.")

    def __len__(self):
        return len(self.da_test_in)

    def __getitem__(self, idx):
        return self.da_test_in_tensor[idx], self.da_test_tar_tensor[idx]

