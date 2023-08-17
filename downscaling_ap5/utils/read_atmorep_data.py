# SPDX-FileCopyrightText: 2023 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2023-05-21"
__update__ = "2023-08-15"

# import modules
import os, sys
sys.path.append("./")
import glob
import json
from typing import List, Tuple
from time import time as timer

import pandas as pd
import xarray as xr
import numpy as np

from other_utils import doy_to_mo, shape_from_str

# the class to handle AtmoRep data
class HandleAtmoRepData(object):
    """
    Handle outout data of AtmoRep.
    TO-DO:
        - get dx-info from token-config
        - add ibatch-information to coordinates of masked data
        - sanity check on data (partly done)
    """
    known_data_types = ["source", "prediction", "target", "ensembles"]

    # offsets to get reconstruct correct timestamps
    days_offset = 1
    hours_offset = 0
    
    def __init__(self, model_id: str, dx_in: float, atmorep_dir: str = "/p/scratch/atmo-rep/results/", 
                 in_dir: str = "/p/scratch/atmo-rep/data/era5/ml_levels/",
                 target_type: str = "fields_prediction", dx_tar: float = None,
                 tar_dir: str = None, epsilon: float = 0.001):
        """
        :param model_id: ID of Atmorep-run to process
        :param dx_in: grid spacing of input data
        :param atmorep_dir: Base-directory where AtmoRep-output is located
        :param in_dir: Base-directory for input data used to train AtmoRep (to get normalization/correction-data)
        :param target_type: Either fields_prediction or fields_target
        :param dx_tar: grid spacing of target data
        :param tar_dir: Base-directory for target data used to train AtmoRep (to get normalization/correction-data)
        :param epsilon: epsilon paramter for log transformation (if applied at all)
        """
        self.model_id = model_id if model_id.startswith("id") else f"id{model_id}"
        self.datadir = os.path.join(atmorep_dir, self.model_id)
        self.datadir_input = in_dir
        self.datadir_target = self.datadir_input if tar_dir is None else tar_dir
        self.target_type = target_type
        self.dx_in = dx_in
        self.dx_tar = self.dx_in if dx_tar is None else dx_tar
        self.epsilon = epsilon
        self.config_file, self.config = self._get_config()
        self.input_variables = self._get_invars()
        self.target_variables = self._get_tarvars()
        
        self.input_token_config = self.get_input_token_config()
        self.target_token_config = self.get_target_token_config() 
        
    def _get_config(self) -> Tuple[str, dict]:
        """
        Get configuration dictionary of trained AtmoRep-model.
        """
        config_jsf = os.path.join(self.datadir, f"model_{self.model_id}.json")
        with open(config_jsf) as json_file:
            config = json.load(json_file)
        return config_jsf, config
    
    def _get_invars(self) -> List:
        """
        Get list of input variables of trained AtmoRep-model.
        """
        return [var_list[0] for var_list in self.config["fields"]]
        #return list(np.asarray(self.config["fields"], dtype=object)[:, 0])
    
    def _get_tarvars(self) -> List:
        """
        Get list of target variables of trained AtmoRep-model.
        """
        return [var_list[0] for var_list in self.config[self.target_type]]
        #return list(np.asarray(self.config[self.target_type])[:, 0])
    
    def _get_token_config(self, key) -> dict:
        """
        Generic function to retrieve token configuration.
        :param key: key-string from which token-info is deducable
        :return dictionary of token info
        """
        token_config_keys = ["general_config", "vlevel", "num_tokens", "token_shape", "bert_parameters"]
        token_config = {var[0]: None for var in self.config[key]}        
        for i, var in enumerate(token_config):
            len_config = len(self.config[key][i])
            token_config[var] =  {config_key: self.config[key][i][j+1] for j, config_key in enumerate(token_config_keys)}
            if len_config >= 7:
                token_config[var]["norm_type"] = self.config[key][i][6]
                if len_config >= 8:
                    if isinstance(self.config[key][i][7][-1], bool):    
                        token_config[var]["log_transform"] = self.config[key][i][7][-1]
                    else: 
                        token_config[var]["log_transform"] = False
            else: # default setting for normalization type
                token_config[var]["norm_type"] = "global"
        
        return token_config
    
    def get_input_token_config(self) -> dict:
        """
        Get input token configuration
        """
        return self._get_token_config("fields")
        
    def get_target_token_config(self) -> dict:
        """
        Retrieve token configuration of output/target data.
        Note that the token configuration is the same as the input data as long as target_fields is unset.
        """
        if self.target_type in ["target_fields", "fields_targets"]:
            return self._get_token_config(self.target_type)   
        else:
            return self.input_token_config
    
    def _get_token_info(self, token_type: str, rank: int = 0, epoch: int = 0, batch: int = 0, varname: str = None, lmasked: bool = True):  
        """
        Retrieve token information. 
        Note: For BERT-training (i.e. lmasked == True), the tokens are scattered in time and space. 
              Thus, the token information will be returned as an index-based array, whereas the token information
              is returned in a structured manner (i.e. shaped as (nbatch, nvlevel, nt, ny, nx) where nt, ny, nx 
              represent the number of tokens in (time, lat, lon)-dimension.
        :param token_type: Type of token for which info should be retrieved (either 'input', 'target', 'ensemble' or 'prediction')
        :param rank: rank (of job) that has created the requested token information file
        :param epoch: training epoch of requested token information file
        :param batch: batch of requested token information
        :param varname: name of variable for which token info is requested
        :param lmasked: flag if masking was applied on token (for token_type 'target' or 'prediction' only)
        """
        if self.dx_tar != self.dx_in:   # in case of different grid spacing in target and source data, target specific token info files are present
            add_str_tar = "target"
            # ML: Required due to inconsistency in naming convetion for files providing token info
            #     and masked token indices (see below)
            add_str_tar2 = "targets" 
        else:
            add_str_tar, add_str_tar2 = "", ""
        var_aux = varname
        if token_type == self.known_data_types[0]:
            var_aux = self.input_variables[0] if var_aux is None else var_aux      
            fpatt = os.path.join(self.datadir, f"*rank{rank}_epoch{epoch:05d}_batch{batch:05d}_token_infos_{var_aux}*.dat")
            token_dict = self.input_token_config[var_aux]
        elif token_type in self.known_data_types[1:]:
            var_aux = self.target_variables[0] if var_aux is None else var_aux 
            fpatt = os.path.join(self.datadir, f"*rank{rank}_epoch{epoch:05d}_batch{batch:05d}_{add_str_tar}_token_infos_{var_aux}*.dat")
            token_dict = self.target_token_config[var_aux]
        else:
            raise ValueError(f"Parsed token type '{token_type}' is unknown. Choose one of the following: {*self.known_data_types,}")
        # Get file for token info
        fname_tokinfos_now = self.get_token_file(fpatt)
        
        # read token info and reshape
        shape_aux = shape_from_str(fname_tokinfos_now)
        tokinfo_data = np.fromfile(fname_tokinfos_now, dtype=np.float32)
        # hack for downscaling
        if token_type in ["prediction", "target"] and "downscaling_num_layers" in self.config:
            token_dict["num_tokens"] = [1, *token_dict["num_tokens"][1::]]
        tokinfo_data = tokinfo_data.reshape(shape_aux[0], len(token_dict["vlevel"]), *token_dict["num_tokens"], shape_aux[-1])

        if token_type in self.known_data_types[1:] and lmasked:
            # read indices of masked tokens
            fpatt = os.path.join(self.datadir, f"*rank{rank}_epoch{epoch:05d}_batch{batch:05d}_{add_str_tar2}_tokens_masked_idx_{var_aux}*.dat")
            fname_tok_maskedidx = self.get_token_file(fpatt)
            
            tok_masked_idx = np.fromfile(fname_tok_maskedidx, dtype=np.int64)
            # reshape token info for slicing...
            tokinfo_data = tokinfo_data.transpose(1, 0, 2, 3, 4, 5).reshape( -1, shape_aux[-1])
            # ... and slice to relevant data
            tokinfo_data = tokinfo_data[tok_masked_idx]   
        
        return tokinfo_data
    
    def _get_date_nomask(self, tokinfo_data, nt):
        
        assert tokinfo_data.ndim == 6, f"Parsed token info data has {tokinfo_data.ndim} dimensions, but 6 are expected."
        
        times = np.array(np.floor(np.delete(tokinfo_data, np.s_[3:], axis = -1)), dtype = np.int32) # ?
        times = times[:,0,:,0,0,:]                                     # remove redundant dimensions

        years, days, hours = times[:,:,0].flatten(), times[:,:,1].flatten(), times[:,:,2].flatten()
        years = np.array([int(y) for y in years])

        # ML: Should not be required in the future!!!
        if any(days < 0):
            years = np.where(days < 0, years - 1, years)
            days_new = np.array([pd.Timestamp(y, 12, 31).dayofyear - self.days_offset for y in years])
            days = np.where(days < 0, days_new - days, days)

        if any(days > 364):
            days_per_year = np.array([pd.Timestamp(y, 12, 31).dayofyear for y in years]) - self.days_offset
            years = np.where(days > days_per_year, years + 1, years)
            days = np.where(days > days_per_year, days - days_per_year - self.days_offset, days)

        # construct date information for centered data position of tokens
        dates = doy_to_mo(days + self.days_offset, years)
        dates = dates + pd.TimedeltaIndex(hours.flatten(), unit='h')
        ### ML: Is this really required???
        # appy hourly offset
        dates = dates - pd.TimedeltaIndex(np.ones(dates.shape)*self.hours_offset, unit="h")
        # reshape and construct remaining date information of tokens
        dates = np.array(dates).reshape(times.shape[0:-1])
        dates = np.array([list(dates + pd.TimedeltaIndex(np.ones(dates.shape)*hh, unit="h")) for hh in range(-int(nt/2), int(nt/2) + 1)])
        dates = dates.transpose(1, 2, 0).reshape(times.shape[0], -1)

        return dates

        # times = np.array(np.floor(np.delete(tokinfo_data, np.s_[3:], axis = -1)), dtype = np.int32) # ?
        # times = times[:,0,:,0,0,:]                                     # remove redundant dimensions

        # years, days, hours = times[:,:,0].flatten(), times[:,:,1].flatten(), times[:,:,2].flatten()
        # years = np.array([int(y) for y in years])
        
        # # ML: Should not be required in the future!!!
        # if any(days < 0):
        #     years = np.where(days < 0, years - 1, years)
        #     days_new = np.array([pd.Timestamp(y, 12, 31).dayofyear - self.days_offset for y in years])
        #     days = np.where(days < 0, days_new - days, days)

        # if any(days > 364):
        #     days_per_year = np.array([pd.Timestamp(y, 12, 31).dayofyear for y in years]) - self.days_offset
        #     years = np.where(days > days_per_year, years + 1, years)
        #     days = np.where(days > days_per_year, days - days_per_year - self.days_offset, days)
        
        # # construct date information for centered data position of tokens
        # dates = doy_to_mo(days + self.days_offset, years)
        # dates = dates + pd.TimedeltaIndex(hours.flatten(), unit='h')
        # ### ML: Is this really required???
        # dates = dates - pd.TimedeltaIndex(np.ones(dates.shape)*self.hours_offset, unit="h")
        
        # # reshape and construct remaining date information of tokens
        # dates = np.array(dates).reshape(times.shape[0:-1]) 
        # dates = np.array([dates + pd.TimedeltaIndex(np.ones(dates.shape)*hh, unit="h") for hh in range(-int(nt/2), int(nt/2) + 1)])
        # print(dates.shape)
        # print(times.shape[0])
        # dates = dates.transpose(1, 2, 0).reshape(times.shape[0], -1)
        
        # return dates
    
    def _get_date_masked(self, tokinfo_data, nt):
        
        assert tokinfo_data.ndim == 2, f"Parsed token info data has {tokinfo_data.ndim} dimensions, but 2 are expected."
        
        years, days, hours = tokinfo_data[:, 0].flatten(), tokinfo_data[:, 1].flatten(), tokinfo_data[:, 2].flatten()
        days = np.array(np.floor(days), dtype=np.int32)
        # ML: Should not be required in the future!!!
        if any(days < 0):
            years = np.where(days < 0, years - 1, years)
            days_new = np.array([pd.Timestamp(y, 12, 31).dayofyear - self.days_offset for y in years])
            days = np.where(days < 0, days_new - days, days)

        if any(days > 364):
            days_per_year = np.array([pd.Timestamp(y, 12, 31).dayofyear for y in years]) - self.days_offset
            years = np.where(days > days_per_year, years + 1, years)
            days = np.where(days > days_per_year, days - days_per_year - self.days_offset, days)
                          
        dates = doy_to_mo(days + self.days_offset, years)      # add 1 to days since counting starts with zero in token_info
        dates = dates + pd.TimedeltaIndex(hours, unit='h')
        ### ML: Is this really required???
        # appy hourly offset
        dates = dates - pd.TimedeltaIndex(np.ones(dates.shape)*self.hours_offset, unit="h") 
        
        # reshape and construct remaining date information of tokens  
        dates = np.array([dates + pd.TimedeltaIndex(np.ones(dates.shape)*hh, unit="h") for hh in range(-int(nt/2), int(nt/2) + 1)])
        dates = dates.transpose()
        
        return np.array(dates)  
        
    def get_date(self, tokinfo_data, token_config):
        """
        Retrieve dates from token info data 
        :param tokinfo_data: token info data which was read beforehand by _get_token_info-method
        :param token_config: corresponding token configuration
        """
        nt = token_config["token_shape"][0]
        
        ndims_tokinfo = tokinfo_data.ndim
        
        if ndims_tokinfo == 2:
            get_date_func = self._get_date_masked
        elif ndims_tokinfo == 6:
            get_date_func = self._get_date_nomask
        else:
            raise ValueError(f"Parsed tokinfo_data-array has unexpected number of dimensions ({ndims_tokinfo}).")
        
        dates = get_date_func(tokinfo_data, nt)       
        
        return dates
    
    def get_grid(self, tokinfo_data, token_config, dx):
        """
        Retrieve underlying geo/grid information.
        :param tokinfo_data: token info data which was read beforehand by _get_token_info-method
        :param token_config: corresponding token configuration
        :param dx: spacing of underlying grid
        """
        ndims_tokinfo = tokinfo_data.ndim
        
        if ndims_tokinfo == 2:
            get_grid_func = self._get_grid_masked
        elif ndims_tokinfo == 6:
            get_grid_func = self._get_grid_nomask
        else:
            raise ValueError(f"Parsed tokinfo_data-array has unexpected number of dimensions ({ndims_tokinfo}).")
            
        lats, lons = get_grid_func(tokinfo_data, token_config, dx)
        
        return lats, lons
    
    
    def read_one_file(self, fname: str, token_type: str, varname: str, token_config: dict, token_info, dx: float, lmasked: bool = True, ldenormalize: bool = True,
                     no_mean_denorm: bool = False):
        """
        Read data from a single output file of AtmoRep and convert to xarray DataArray with underlying coordinate information.
        :param token_type: Type of token for which info should be retrieved (either 'input', 'target', 'ensemble' or 'prediction')
        :param rank: rank (of job) that has created the requested token information file
        :param epoch: training epoch of requested token information file
        :param batch: batch of requested token information
        :param varname: name of variable for which token info is requested
        :param lmasked: flag if masking was applied on token (for token_type 'target' or 'prediction' only)
        :param ldenormalize: flag if denormalize/invert correction should be applied (also includes inversion of log transformation)
        :param no_mean_denorm: flag if mean should not be added when denormalization is performed
        """              
        data = np.fromfile(fname, dtype=np.float32)
        
        times = self.get_date(token_info, token_config)
        lats, lons = self.get_grid(token_info, token_config, dx)
        
        if token_type in self.known_data_types[1:] and lmasked:
            data = self._reshape_masked_data(data, token_config, token_type == "ensembles", self.config["net_tail_num_nets"])
            vlvl = np.array(token_info[:, 3], dtype=np.int32)
            data = self.masked_data_to_xarray(data, varname, times, vlvl, lats, lons, token_type == "ensembles")
        else:
            data = self._reshape_nomask_data(data, token_config, self.config["batch_size_test"])
            data = self.nomask_data_to_xarray(data, varname, times, token_config["vlevel"], lats, lons)

        if ldenormalize:
            t0 = timer()

            if getattr(token_config, "log_transform", False):
                data = self.invert_log_transform(data)

            if token_type in self.known_data_types[1:] and lmasked:
                data = self.denormalize_masked_data(data, token_type, token_config["norm_type"], no_mean_denorm)
            else:
                data = self.denormalize_nomask_data(data, token_type, token_config["norm_type"], no_mean_denorm)   
                
            data.attrs["denormalization time [s]"] = timer() - t0
        
        return data
    
    def read_data(self, token_type: str, varname: str, rank: int = -1, epoch: int = -1, batch: int = -1, lmasked: bool = True,
                  ldenormalize: bool = True, no_mean_denorm: bool = False):
        """
        Read data from a single output file of AtmoRep and convert to xarray DataArray with underlying coordinate information.
        :param token_type: Type of token for which info should be retrieved (either 'input', 'target', 'ensemble' or 'prediction')
        :param rank: rank (of job) that has created the requested token information file
        :param epoch: training epoch of requested token information file
        :param batch: batch of requested token information
        :param varname: name of variable for which token info is requested
        :param lmasked: flag if masking was applied on token (for token_type 'target' or 'prediction' only)
        :param ldenormalize: flag if denormalize/invert correction should be applied (also includes inversion of log transformation)
        :param no_mean_denorm: flag if mean should not be added when denormalization is performed
        """      
        if token_type == "source":
            token_type_str = "source"
            token_config = self.input_token_config[varname]
            dx = self.dx_in
        elif token_type in self.known_data_types[1:]:
            token_type_str = "preds" if token_type == "prediction" else token_type
            token_config = self.target_token_config[varname]
            dx = self.dx_tar
        else:
            raise ValueError(f"Parsed token type '{token_type}' is unknown. Choose one of the following: {*self.known_data_types,}")
            
        filelist = self.get_hierarchical_sorted_files(token_type_str, varname, rank, epoch, batch)
        
        print(f"Start reading {len(filelist)} files for {token_type} data...")
        lwarn = True
        for i, f in enumerate(filelist):
            rank, epoch, batch = self.get_rank_epoch_batch(f)
            try:
                token_info = self._get_token_info(token_type, rank=rank, epoch=epoch, batch=batch, varname=varname, lmasked=lmasked)
            except FileNotFoundError:
                if lwarn:   # print warning only once
                    print(f"No token info for {token_type} data of {varname} found. Proceed with token info for input data.")
                    lwarn = False
                token_info = self._get_token_info(self.known_data_types[0], rank=rank, epoch=epoch, batch=batch, varname=None, lmasked=lmasked)
                
            da_f = self.read_one_file(f, token_type, varname, token_config, token_info, dx, lmasked, ldenormalize, no_mean_denorm)
            
            if i == 0:
                da = da_f.copy()
            else:
                dim_concat = da.dims[0]
                ilast = da[dim_concat][-1] + 1
                inew = np.arange(ilast, ilast + len(da_f[dim_concat]))
                da_f = da_f.assign_coords({dim_concat: inew})
                if ldenormalize:
                    da.attrs["denormalization time [s]"] += da_f.attrs["denormalization time [s]"]
                                   
                da = xr.concat([da, da_f], dim=da.dims[0])

        if ldenormalize:
            print(f"Denormalization of {len(filelist)} files for {token_type} data took {da.attrs['denormalization time [s]']:.2f}s")
                
        return da
        
    def get_hierarchical_sorted_files(self, token_type_str: str, varname: str, rank: int = -1, epoch: int = -1, batch: int = -1):
        rank_str = f"rank*" if rank == -1 else f"rank{rank:d}"
        epoch_str = f"epoch*" if epoch == -1 else f"epoch{epoch:05d}"
        batch_str = f"batch*" if batch == -1 else f"batch{batch:05d}"
        
        fpatt = f"*{self.model_id}_{rank_str}_{epoch_str}_{batch_str}_{token_type_str}_{varname}*.dat"
        filelist = glob.glob(os.path.join(self.datadir, fpatt))

        filelist = [f for f in filelist if '000-1' not in f] #remove epoch -1

        if len(filelist) == 0:
            raise FileNotFoundError(f"Could not file any files mathcing pattern '{fpatt}' under directory '{self.datadir}'.")
        
        # hierarchical sorting: epoch -> rank -> batch
        sorted_filelist = sorted(filelist, key=lambda x: self.get_number(x, "_rank"))
        sorted_filelist = sorted(sorted_filelist, key=lambda x: self.get_number(x, "_epoch"))
        sorted_filelist = sorted(sorted_filelist, key=lambda x: self.get_number(x, "_batch"))
      
        return sorted_filelist
    
    def denormalize_global(self, da, param_dir, no_mean = False):
        
        da_dims = list(da.dims)
        varname = da.name
        vlvl = da["vlevel"].values #list(da["vlevel"].values)[0]
        # get nomralization parameters
        mean, std = self.get_global_norm_params(varname, vlvl, param_dir)
        if no_mean:
            mean[...] = 0.

        # re-index data along time dimension for efficient denormalization
        time_dims = list(da["time"].dims)
        da = da.stack({"time_aux": time_dims})
        time_save = da["time_aux"]                # save information for later re-indexing
        da = da.set_index({"time_aux": "time"}).sortby("time_aux")
        time_save = time_save.sortby("time")

        da = da.resample({"time_aux": "1M"})
        # loop over year-month items
        for i, (ts, da_mm) in enumerate(da):
            yr_mm = pd.to_datetime(ts).strftime("%Y-%m")
            da_mm = da_mm * std.sel({"year_month": yr_mm}).values + mean.sel({"year_month": yr_mm}).values
            if i == 0:
                da_concat = da_mm.copy()
            else:
                da_concat = xr.concat([da_concat, da_mm], dim="time_aux")

        da_concat["time_aux"] = time_save
<<<<<<< HEAD:analysis/utils/read_atmorep_data.py

=======
>>>>>>> c187c1ec (Update of read_atmorep_data.py to supoort downscaling data.):analysis/read_atmorep_data.py
        if not xr.__version__ == "0.20.1":
            da_concat = da_concat.reset_index("time_aux")
        da_concat = da_concat.unstack("time_aux")

        return da_concat.transpose(*da_dims)

    def denormalize_masked_data(self, data: xr.DataArray, token_type: str, norm_type: str, no_mean: bool = False):
        """
        Denormalizes/Inverts correction for masked data.
        Data has to normalized considering vertical level and time which both vary along token-dimension.
        :param data: normalized (xarray) data array providing masked data (cf. masked_data_to_xarray-method) 
        :param token_type: type of token to be handled, e.g. 'source' (cf. known_data_types)
        :param norm_type: type of normalization applied to the data (either 'local' or 'global')
        :param no_mean: flag if data normalization has NOT been zero-meaned 
        """
        mm_yr = np.unique(data["time"].dt.strftime("y%Y_m%m"))
        vlevels = np.unique(data["vlevel"])
        varname = data.name
        dim0 = data.dims[0]
        basedir = self.datadir_input if token_type == self.known_data_types[0] else self.datadir_target

        for vlvl in vlevels:
            if norm_type == "local":
                datadir = os.path.join(basedir, f"{vlvl}", "corrections", varname)
                for month in mm_yr:
                    fcorr_now = os.path.join(datadir, f"corrections_mean_var_{varname}_{month}_ml{vlvl}.nc")
                    norm_data = xr.open_dataset(fcorr_now)
                    mean, std = norm_data[f"{varname}_ml{vlvl:d}_mean"], norm_data[f"{varname}_ml{vlvl:d}_std"]
                    if no_mean: mean[...] = 0.

                    mask = (data["vlevel"] == vlvl) & (data["time"].dt.strftime("y%Y_m%m") == month)
                    data_now = data.where(mask, drop=True)
                    for it in data_now[dim0]:   # loop over all tokens
                        xy_dict = {"lat": data_now.sel({dim0: it})["lat"], "lon": data_now.sel({dim0: it})["lon"]}
                        mu_it, std_it = mean.sel(xy_dict), std.sel(xy_dict)
                        data_now.loc[{dim0: it}] = data_now.loc[{dim0: it}]*std_it + mu_it
                    data = xr.where(mask, data_now, data)
                        
            elif norm_type == "global":
                mask = data["vlevel"] == vlvl
                data = xr.where(mask, self.denormalize_global(data.where(mask), basedir, no_mean = no_mean), data)
                    
        return data
    
    def denormalize_nomask_data(self, data: xr.DataArray, token_type:str, norm_type:str, no_mean: bool = False):
        """
        Denormalizes/Inverts correction for unmasked data.
        Data has to normalized considering vertical level and time which both vary along token-dimension.
        :param data: normalized (xarray) data array providing unmasked data (cf. nomask_data_to_xarray-method) 
        :param token_type: type of token to be handled, e.g. 'source' (cf. known_data_types)
        :param norm_type: type of normalization applied to the data (either 'local' or 'global')
        :param no_mean: flag if data normalization has NOT been zero-meaned 
        """ 
        times = data["time"]
        nt = len(times[0,:])
        dim0 = data.dims[0]
        varname = data.name
        basedir = self.datadir_input if token_type == self.known_data_types[0] else self.datadir_target
        
        center_times = times.min(dim="t") + pd.Timedelta(nt/2, "hours")
        yr_mm = np.unique(center_times.dt.strftime("y%Y_m%m"))

        for vlvl in data["vlevel"]:
            if norm_type == "local":
                iv = vlvl.values 
                data_dir = os.path.join(basedir, f"{iv:d}", "corrections", varname)
                for month in yr_mm:
                    fcorr_now = os.path.join(data_dir, f"corrections_mean_var_{varname}_{month}_ml{iv:d}.nc")
                    norm_data = xr.open_dataset(fcorr_now)
                    mean, std = norm_data[f"{varname}_ml{iv:d}_mean"], norm_data[f"{varname}_ml{iv:d}_std"]
                    if no_mean: mean[...] = 0.

                    mask = (data["time"].dt.strftime("y%Y_m%m") == month)
                    data_now = data.where(mask, drop=True)
                    for it in data_now[dim0]:
                        xy_dict = {"lat": data_now.sel({dim0: it})["lat"], "lon": data_now.sel({dim0: it})["lon"]}
                        mu_it, std_it = mean.sel(xy_dict), std.sel(xy_dict)
                        data.loc[{dim0: it, "vlevel": iv}] = data.loc[{dim0: it, "vlevel": iv}]*std_it + mu_it 

            elif norm_type == "global": 
                data.loc[{"vlevel": vlvl}] = self.denormalize_global(data.sel({"vlevel": vlvl}), basedir, no_mean = no_mean)  
                    
        return data

    def invert_log_transform(self, data):
        """
        Inverts log transformation on data.
        param data: the xarray DataArray which was log transformed
        :return: data after inversion of log transformation
        """
        data = self.epsilon*(np.exp(data) - 1.)
        
        return data
    
    def get_rank_epoch_batch(self, fname, to_int: bool=True):
        rank = self.get_number(fname, "_rank")
        epoch = self.get_number(fname, "_epoch")
        batch = self.get_number(fname, "_batch")
        
        if to_int:
            rank, epoch, batch = int(rank), int(epoch), int(batch)
        
        return rank, epoch, batch
    
    @staticmethod
    def nomask_data_to_xarray(data_np, varname: str, times, vlvls, lat, lon, lensemble: bool = False):
        
        if lensemble:
            raise ValueError("Ensemlbe not supported yet.")
        
        nbatch, nt = data_np.shape[0], data_np.shape[2]
        nlat, nlon = len(lat[0, :]), len(lon[0, :])
        
        da = xr.DataArray(data_np, dims=["ibatch", "vlevel", "t", "y", "x"],
                          coords={"ibatch": np.arange(nbatch), "vlevel": vlvls,
                                  "t": np.arange(nt), "y": np.arange(nlat), "x": np.arange(nlon),
                                  "time": (["ibatch", "t"], times), 
                                  "lat": (["ibatch", "y"], lat), "lon": (["ibatch", "x"], lon)},
                         name=varname)
        return da
    
    @staticmethod
    def masked_data_to_xarray(data_np, varname: str, times, vlvls, lat, lon, lensemble: bool = False):
            
        data_dims = ["itoken", "tt", "yt", "xt"]        
        if lensemble:
            ntoken, nens, ntt, yt, xt = data_np.shape
            data_dims.insert(1, "ens")
        else:
            ntoken, ntt, yt, xt = data_np.shape

        data_coords = {"itoken": np.arange(ntoken), "tt": np.arange(ntt), "yt": np.arange(yt), 
                       "xt": np.arange(xt), "time": (["itoken", "tt"], times),
                       "vlevel": ("itoken", vlvls),
                       "lat": (["itoken", "yt"], lat), "lon": (["itoken", "xt"], lon)}        
        
        if lensemble:
            data_coords["ens"] = np.arange(nens)
        
        da = xr.DataArray(data_np, dims=data_dims, coords=data_coords, name = varname)

        return da
    
    @staticmethod
    def get_number(file_name, split_arg):
        # Extract the number from the file name using the provided split argument
        return int(file_name.split(split_arg)[1].split('_')[0])
    
    @staticmethod
    def get_token_file(fpatt: str, nfiles: int = 1):
        # Get file for token info
        fnames = glob.glob(fpatt)
        # sanity check
        if not fnames:
            raise FileNotFoundError(f"Could not find required file(-s) using the following filename pattern: '{fpatt}'")
            
        assert len(fnames) == nfiles, f"More files matching filename pattern '{fpatt}' found than expected."
        
        if nfiles == 1: 
            fnames = fnames[0]
        
        return fnames    
    
    @staticmethod
    def _get_grid_nomask(tokinfo_data, token_config, dx):
        """
        Retrieve underlying geo/grid information for unmasked data (complete patches!)
        :param tokinfo_data: token info data which was read beforehand by _get_token_info-method
        :param token_config: corresponding token configuration
        :param dx: spacing of underlying grid
        """
        # retrieve spatial token size
        ny, nx = token_config["token_shape"][1:3]

        # off-centering for even number of grid points
        ny1, nx1 = 1, 1
        
        if ny%2 == 0: 
            ny1 = 0
        if nx%2 == 0:
            nx1 = 0
        
        lat_offset = np.arange(-int((ny-ny1)/2)*dx, int(ny/2+ny1)*dx, dx)
        lon_offset = np.arange(-int((nx-nx1)/2)*dx, int(nx/2+nx1)*dx, dx)   
        
        #IMPORTANT: need to swap axes in lats to have ntokens_lat adjacent to tokinfo --> 8x4x8x8 -> 8x8x4x8
        lons  = np.array([tokinfo_data.swapaxes(0, -1)[-3]+lon_offset[i] for i in range(len(lon_offset))]).swapaxes(0, -1)%360
        lats  = np.array([tokinfo_data.swapaxes(-3, -2).swapaxes(0, -1)[-4]+lat_offset[i] for i in range(len(lat_offset))]).swapaxes(0, -1)%180
        # if(flip_lats):
        #     lats  = np.flip(lats)
        
        lats, lons = lats[:, 0, 0, 0, :, :], lons[:, 0, 0, 0, :, :]
        lats, lons = lats.reshape(lats.shape[0], -1), lons.reshape(lats.shape[0], -1)
        
        # correct lat values because they are in 'mathematical coordinates' with 0 at the North Pole
        lats = 90. -lats
        
        return lats, lons  
    
    @staticmethod
    def _get_grid_masked(tokinfo_data, token_config, dx):  
        """
        Retrieve underlying geo/grid information for masked data (scattered tokens!)
        :param tokinfo_data: token info data which was read beforehand by _get_token_info-method
        :param token_config: corresponding token configuration
        :param dx: spacing of underlying grid
        """
        # retrieve spatial token size
        ntok = tokinfo_data.shape[0]
        ny, nx = token_config["token_shape"][1:3]
        
        # off-centering for even number of grid points
        ny1, nx1 = 1, 1
        if ny%2 == 0: 
            ny1 = 0
        if nx%2 == 0:
            nx1 = 0
        
        lat_offset = np.arange(-int((ny-ny1)/2+nx1)*dx, int(ny/2+ny1)*dx, dx)
        lon_offset = np.arange(-int((nx-nx1)/2+ny1)*dx, int(nx/2+nx1)*dx, dx) 
        
        #lats = np.array([tokinfo_data[idx, 4] + np.arange(-int(ny/2)*dx, int(ny/2+1)*dx, dx) for idx in range(ntok)]) % 180 #boundary conditions
        #lons = np.array([tokinfo_data[idx, 5] + np.arange(-int(nx/2)*dx, int(nx/2+1)*dx, dx) for idx in range(ntok)]) % 360 #boundary conditions
        lats = np.array([tokinfo_data[idx, 4] + lat_offset for idx in range(ntok)]) % 180 #boundary conditions
        lons = np.array([tokinfo_data[idx, 5] + lon_offset for idx in range(ntok)]) % 360 #boundary conditions
        
        # correct lat values because they are in 'mathematical coordinates' with 0 at the North Pole
        lats = 90. - lats
        
        return lats, lons
    

    @staticmethod
    def get_global_norm_params(varname, vlv, basedir):
        """
        Read parameter files for global z-score normalization
        :param varname: name of variable
        :param vlv: vertical level index
        :param basedir: base directory under which correction/parameter files are located
        """
        fcorr = os.path.join(basedir, f"{vlv}", "corrections", f"global_corrections_mean_var_{varname}_ml{vlv:d}.bin")
        corr_data = np.fromfile(fcorr, dtype="float32").reshape(-1, 4)

        years, months = corr_data[:,0], corr_data[:,1]

        # hack: filter data where years equal to zero
        bad_inds = np.nonzero(years == 0)
        years, months = np.delete(years, bad_inds), np.delete(months, bad_inds)
        mean, var = np.delete(corr_data[:,2], bad_inds), np.delete(corr_data[:,3], bad_inds)

        yr_mm = [pd.to_datetime(f"{int(yr):d}-{int(m):02d}", format="%Y-%m") for yr, m in zip(years, months)]

        mean = xr.DataArray(mean, dims=["year_month"], coords={"year_month": yr_mm})
        var = xr.DataArray(var, dims=["year_month"], coords={"year_month": yr_mm})

        return mean, var    
    
    @staticmethod
    def _reshape_nomask_data(data, token_config: dict, batch_size: int):
        """
        Reshape unmasked token data that has been read from .dat-files of AtmoRep (complete patches!).
        Data is assumed to cover 
        Adapted from Ilaria, but now with in-place operations to save memory.
        """
        sh = (batch_size, len(token_config["vlevel"]), *token_config["num_tokens"], *token_config["token_shape"])
        data = data.reshape(*sh)
        
        # further reshaping to collapse token dimensions 
        # The following is adapted from Ilaria, but now with in-place operations to save memory (original code, see below).
        data = np.transpose(data, (0,1,2,5,3,6,4,7))
        data = data.reshape(*data.shape[:-2], -1)
        data = data.reshape(*data.shape[:-3], -1, *data.shape[-1:])
        data = data.reshape(*data.shape[:2], -1, *data.shape[4:])

        return data
    
    @staticmethod
    def _reshape_masked_data(data, token_config, lensemble: bool = False, nens: int = None):
        sh0 = (-1,)
        if lensemble:
            assert nens > 0, f"Invalid nens value passed ({nens}). It must be an integer > 0"
            sh0 = (-1, nens)

        token_sh = token_config["token_shape"]
        data = data.reshape(*sh0, *token_sh)
        
        return data
