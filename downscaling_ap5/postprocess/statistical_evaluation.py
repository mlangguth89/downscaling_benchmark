# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Collection of auxiliary functions for statistical evaluation and class for Score-functions
"""

__email__ = "b.gong@fz-juelich.de"
__author__ = "Michael Langguth"
__date__ = "2021-05-xx"

import numpy as np
import xarray as xr
from typing import Union, List
from skimage.metrics import structural_similarity as ssim
import datetime
import pandas as pd
try:
    from tqdm import tqdm
    l_tqdm = True
except:
    l_tqdm = False
from general_utils import provide_default, check_str_in_list

# basic data types
da_or_ds = Union[xr.DataArray, xr.Dataset]


def calculate_cond_quantiles(data_fcst: xr.DataArray, data_ref: xr.DataArray, factorization="calibration_refinement",
                             quantiles=(0.05, 0.5, 0.95)):
    """
    Calculate conditional quantiles of forecast and observation/reference data with selected factorization
    :param data_fcst: forecast data array
    :param data_ref: observational/reference data array
    :param factorization: factorization: "likelihood-base_rate" p(m|o) or "calibration_refinement" p(o|m)-> default
    :param quantiles: conditional quantiles
    :return quantile_panel: conditional quantiles of p(m|o) or p(o|m)
    """
    method = calculate_cond_quantiles.__name__

    # sanity checks
    if not isinstance(data_fcst, xr.DataArray):
        raise ValueError("%{0}: data_fcst must be a DataArray.".format(method))

    if not isinstance(data_ref, xr.DataArray):
        raise ValueError("%{0}: data_ref must be a DataArray.".format(method))

    if not (list(data_fcst.coords) == list(data_ref.coords) and list(data_fcst.dims) == list(data_ref.dims)):
        raise ValueError("%{0}: Coordinates and dimensions of data_fcst and data_ref must be the same".format(method))

    nquantiles = len(quantiles)
    if not nquantiles >= 3:
        raise ValueError("%{0}: quantiles must be a list/tuple of at least three float values ([0..1])".format(method))

    if factorization == "calibration_refinement":
        data_cond = data_fcst
        data_tar = data_ref
    elif factorization == "likelihood-base_rate":
        data_cond = data_ref
        data_tar = data_fcst
    else:
        raise ValueError("%{0}: Choose either 'calibration_refinement' or 'likelihood-base_rate' for factorization"
                         .format(method))

    # get and set some basic attributes
    data_cond_longname = provide_default(data_cond.attrs, "longname", "conditioning_variable")
    data_cond_unit = provide_default(data_cond.attrs, "unit", "unknown")

    data_tar_longname = provide_default(data_tar.attrs, "longname", "target_variable")
    data_tar_unit = provide_default(data_cond.attrs, "unit", "unknown")

    # get bins for conditioning
    data_cond_min, data_cond_max = np.floor(np.min(data_cond)), np.ceil(np.max(data_cond))
    bins = list(np.arange(int(data_cond_min), int(data_cond_max) + 1))
    bins_c = 0.5 * (np.asarray(bins[0:-1]) + np.asarray(bins[1:]))
    nbins = len(bins) - 1

    # get all possible bins from target and conditioning variable
    data_all_min, data_all_max = np.minimum(data_cond_min, np.floor(np.min(data_tar))),\
                                 np.maximum(data_cond_max, np.ceil(np.max(data_tar)))
    bins_all = list(np.arange(int(data_all_min), int(data_all_max) + 1))
    bins_c_all = 0.5 * (np.asarray(bins_all[0:-1]) + np.asarray(bins_all[1:]))
    # initialize quantile data array
    quantile_panel = xr.DataArray(np.full((len(bins_c_all), nquantiles), np.nan),
                                  coords={"bin_center": bins_c_all, "quantile": list(quantiles)},
                                  dims=["bin_center", "quantile"],
                                  attrs={"cond_var_name": data_cond_longname, "cond_var_unit": data_cond_unit,
                                         "tar_var_name": data_tar_longname, "tar_var_unit": data_tar_unit})
    
    print("%{0}: Start caclulating conditional quantiles for all {1:d} bins.".format(method, nbins))
    # fill the quantile data array
    for i in np.arange(nbins):
        # conditioning of ground truth based on forecast
        data_cropped = data_tar.where(np.logical_and(data_cond >= bins[i], data_cond < bins[i + 1]))
        # quantile-calculation
        quantile_panel.loc[dict(bin_center=bins_c[i])] = data_cropped.quantile(quantiles)

    return quantile_panel, data_cond


def avg_metrics(metric: da_or_ds, dim_name: str):
    """
    Averages metric over given dimension
    :param metric: DataArray or Dataset of metric that should be averaged
    :param dim_name: name of the dimension on which division into blocks is applied
    :return: DataArray or Dataset of metric averaged over given dimension. If a Dataset is passed, the averaged metrics
             carry the suffix "_avg" in their variable names.
    """
    method = avg_metrics.__name__

    if not isinstance(metric, da_or_ds.__args__):
        raise ValueError("%{0}: Input metric must be a xarray DataArray or Dataset and not {1}".format(method,
                                                                                                       type(metric)))

    if isinstance(metric, xr.Dataset):
        list_vars = [varname for varname in metric.data_vars if dim_name in metric[varname].dims]
        if not list_vars:
            raise ValueError("%{0}: {1} is not a dimension in the input metric dataset".format(method, dim_name))

        metric2avg = metric[list_vars]
    else:
        if dim_name in metric.dims:
            pass
        else:
            raise ValueError("%{0}: {1} is not a dimension in the input metric data-array".format(method, dim_name))

        metric2avg = metric

    metric_avg = metric2avg.mean(dim=dim_name)

    if isinstance(metric, xr.Dataset):
        new_varnames = ["{0}_avg".format(var) for var in list_vars]
        metric_avg = metric_avg.rename(dict(zip(list_vars, new_varnames)))

    return metric_avg


def perform_block_bootstrap_metric(metric: da_or_ds, dim_name: str, block_length: int, nboots_block: int = 1000,
                                   seed: int = 42):
    """
    Performs block bootstrapping on metric along given dimension (e.g. along time dimension)
    :param metric: DataArray or dataset of metric that should be bootstrapped
    :param dim_name: name of the dimension on which division into blocks is applied
    :param block_length: length of block (index-based)
    :param nboots_block: number of bootstrapping steps to be performed
    :param seed: seed for random block sampling (to be held constant for reproducability)
    :return: bootstrapped version of metric(-s)
    """

    method = perform_block_bootstrap_metric.__name__

    if not isinstance(metric, da_or_ds.__args__):
        raise ValueError("%{0}: Input metric must be a xarray DataArray or Dataset and not {1}".format(method,
                                                                                                       type(metric)))
    if dim_name not in metric.dims:
        raise ValueError("%{0}: Passed dimension cannot be found in passed metric.".format(method))

    metric = metric.sortby(dim_name)

    dim_length = np.shape(metric.coords[dim_name].values)[0]
    nblocks = int(np.floor(dim_length/block_length))

    if nblocks < 10:
        raise ValueError("%{0}: Less than 10 blocks are present with given block length {1:d}."
                         .format(method, block_length) + " Too less for bootstrapping.")

    # precompute metrics of block
    for iblock in np.arange(nblocks):
        ind_s, ind_e = iblock * block_length, (iblock + 1) * block_length
        metric_block_aux = metric.isel({dim_name: slice(ind_s, ind_e)}).mean(dim=dim_name)
        if iblock == 0:
            metric_val_block = metric_block_aux.expand_dims(dim={"iblock": 1}, axis=0).copy(deep=True)
        else:
            metric_val_block = xr.concat([metric_val_block, metric_block_aux.expand_dims(dim={"iblock": 1}, axis=0)],
                                         dim="iblock")

    metric_val_block["iblock"] = np.arange(nblocks)

    # get random blocks
    np.random.seed(seed)
    iblocks_boot = np.sort(np.random.randint(nblocks, size=(nboots_block, nblocks)))

    print("%{0}: Start block bootstrapping...".format(method))
    iterator_b = np.arange(nboots_block)
    if l_tqdm:
        iterator_b = tqdm(iterator_b)
    for iboot_b in iterator_b:
        metric_boot_aux = metric_val_block.isel(iblock=iblocks_boot[iboot_b, :]).mean(dim="iblock")
        if iboot_b == 0:
            metric_boot = metric_boot_aux.expand_dims(dim={"iboot": 1}, axis=0).copy(deep=True)
        else:
            metric_boot = xr.concat([metric_boot, metric_boot_aux.expand_dims(dim={"iboot": 1}, axis=0)], dim="iboot")

    # set iboot-coordinate
    metric_boot["iboot"] = np.arange(nboots_block)
    if isinstance(metric_boot, xr.Dataset):
        new_varnames = ["{0}_bootstrapped".format(var) for var in metric.data_vars]
        metric_boot = metric_boot.rename(dict(zip(metric.data_vars, new_varnames)))

    return metric_boot


class Scores:
    """
    Class to calculate scores and skill scores.
    """
    def __init__(self, score_name: str, dims: List[str]):
        """
        Initialize score instance.
        :param score_name: name of score that is queried
        :param dims: list of dimension over which the score shall operate
        :return: Score instance
        """
        method = Scores.__init__.__name__
        self.metrics_dict = {"mse": self.calc_mse_batch , "psnr": self.calc_psnr_batch,
                             "ssim": self.calc_ssim_batch, "acc": self.calc_acc_batch,
                             "texture": self.calc_spatial_variability}
        self.score_name = self.set_score_name(score_name)
        self.score_func = self.metrics_dict[score_name]
        # attributes set when run_calculation is called
        self.avg_dims = dims

    def set_score_name(self, score_name):

        method = Scores.set_score_name.__name__

        if score_name in self.metrics_dict.keys():
            return score_name
        else:
            print("The following scores are currently implemented:".format(method))
            for score in self.metrics_dict.keys():
                print("* {0}".format(score))
            raise ValueError("%{0}: The selected score '{1}' cannot be selected.".format(method, score_name))

    def calc_mse_batch(self, data_fcst, data_ref, **kwargs):
        """
        Calculate mse of forecast data w.r.t. reference data
        :param data_fcst: forecasted data (xarray with dimensions [batch, lat, lon])
        :param data_ref: reference data (xarray with dimensions [batch, lat, lon])
        :return: averaged mse for each batch example, [batch,fore_hours]
        """
        method = Scores.calc_mse_batch.__name__

        if kwargs:
            print("%{0}: Passed keyword arguments are without effect.".format(method))
        # sanity checks
        if self.avg_dims is None:
            print("%{0}: Squared difference is averaged over all dimensions.".format(method))
            dims = list(data_fcst.dims)
        else:
            dims = self.avg_dims

        mse = np.square(data_fcst - data_ref).mean(dim=dims)

        return mse

    def calc_psnr_batch(self, data_fcst, data_ref, **kwargs):
        """
        Calculate psnr of forecast data w.r.t. reference data
        :param data_fcst: forecasted data (xarray with dimensions [batch,fore_hours, lat, lon])
        :param data_ref: reference data (xarray with dimensions [batch, fore_hours, lat, lon])
        :return: averaged psnr for each batch example [batch, fore_hours]
        """
        method = Scores.calc_psnr_batch.__name__

        if "pixel_max" in kwargs:
            pixel_max = kwargs.get("pixel_max")
        else:
            pixel_max = 1.

        mse = self.calc_mse_batch(data_fcst, data_ref)
        if np.count_nonzero(mse) == 0:
            psnr = mse
            psnr[...] = 100.
        else:
            psnr = 20.*np.log10(pixel_max / np.sqrt(mse))

        return psnr

    def calc_ssim_batch(self, data_fcst, data_ref, **kwargs):
        """
        Calculate ssim ealuation metric of forecast data w.r.t reference data
        :param data_fcst: forecasted data (xarray with dimensions [batch, fore_hours, lat, lon])
        :param data_ref: reference data (xarray with dimensions [batch, fore_hours, lat, lon])
        :return: averaged ssim for each batch example, shape is [batch,fore_hours]
        """
        method = Scores.calc_ssim_batch.__name__
        batch_size = np.array(data_ref).shape[0]
        fore_hours = np.array(data_fcst).shape[1]
        ssim_pred = [[ssim(data_ref[i,j, ...],data_fcst[i,j,...]) for j in range(fore_hours)]
                     for i in range(batch_size)]
        return ssim_pred

    def calc_acc_batch(self, data_fcst, data_ref,  **kwargs):
        """
        Calculate acc ealuation metric of forecast data w.r.t reference data
        :param data_fcst: forecasted data (xarray with dimensions [batch, fore_hours, lat, lon])
        :param data_ref: reference data (xarray with dimensions [batch, fore_hours, lat, lon])
        :param data_clim: climatology data (xarray with dimensions [monthly, hourly, lat, lon])
        :return: averaged acc for each batch example [batch, fore_hours]
        """
        method = Scores.calc_acc_batch.__name__
        if "data_clim" in kwargs:
            data_clim = kwargs["data_clim"]
        else:
            raise KeyError("%{0}: climatological data must be parsed to calculate the ACC.".format(method))        

        batch_size = data_fcst.shape[0]
        fore_hours = data_fcst.shape[1]
        acc = np.ones([batch_size,fore_hours])*np.nan
        for i in range(batch_size):
            for j in range(fore_hours):
                img_fcst = data_fcst[i, j, ...]
                img_ref = data_ref[i, j, ...]
                # get the forecast time
                fcst_time = xr.Dataset({'time': pd.to_datetime(img_fcst.init_time.data) + datetime.timedelta(hours=j)})
                img_month = fcst_time.time.dt.month
                img_hour = fcst_time.time.dt.hour
                img_clim = data_clim.sel(month=img_month, hour=img_hour)               
 
                ### HAVE TO SELECT FORM CLIMATE DATA DIRECTLY; done
                #time_idx = (img_month-1)*24+img_hour
                #img_clim = data_clim[time_idx,:,:] 
           
                img1_ = img_ref - img_clim
                img2_ = img_fcst - img_clim
                cor1 = np.sum(img1_*img2_)
                cor2 = np.sqrt(np.sum(img1_**2)*np.sum(img2_**2))
                acc[i, j] = cor1/cor2
        return acc

    def calc_spatial_variability(self, data_fcst, data_ref, **kwargs):
        """
        Calculates the ratio between the spatial variability of differental operator with order 1 (or 2) forecast and
        reference data
        :param data_fcst: data_fcst: forecasted data (xarray with dimensions [batch, fore_hours, lat, lon])
        :param data_ref: reference data (xarray with dimensions [batch, fore_hours, lat, lon])
        :param kwargs: order to control the order of spatial differential operator, 'non_spatial_avg_dims' to perform
                       averaging
        :return: the ratio between spatial variabilty in the forecast and reference data field
        """

        method = Scores.calc_spatial_variability.__name__

        if self.avg_dims is None:
            pass
        else:
            print("%{0}: Passed dimensions to Scores-object instance are ignored.".format(method) +
                  "Make use of 'non_spatial_avg_dims' to pass a list over dimensions for averaging")

        if "order" in kwargs:
            order = kwargs.get("order")
        else:
            order = 1
         
        if "non_spatial_avg_dims" in kwargs:
            add_avg_dims = kwargs.get("non_spatial_avg_dims")
        else:
            add_avg_dims = None

        fcst_grad = Scores.calc_geo_spatial_diff(data_fcst, order=order)
        ref_grd = Scores.calc_geo_spatial_diff(data_ref, order=order)

        ratio_spat_variability = fcst_grad/ref_grd

        if add_avg_dims: ratio_spat_variability = ratio_spat_variability.mean(dim=add_avg_dims)

        return ratio_spat_variability

    @staticmethod
    def calc_geo_spatial_diff(scalar_field: xr.DataArray, order: int = 1, r_e: float = 6371.e3, avg_dom: bool = True):
        """
        Calculates the amplitude of the gradient (order=1) or the Laplacian (order=2) of a scalar field given on a regular,
        geographical grid (i.e. dlambda = const. and dphi=const.)
        :param scalar_field: scalar field as data array with latitude and longitude as coordinates
        :param order: order of spatial differential operator
        :param r_e: radius of the sphere
        :param avg_dom: flag if amplitude is averaged over the domain
        :return: the amplitude of the gradient/laplacian at each grid point or over the whole domain (see avg_dom)
        """
        method = Scores.calc_geo_spatial_diff.__name__

        # sanity checks
        assert isinstance(scalar_field, xr.DataArray), "%{0}: scalar_field must be a xarray DataArray."\
                                                       .format(method)
        assert order in [1, 2], "%{0}: Order must be either 1 or 2.".format(method)

        dims = list(scalar_field.dims)
        lat_dims = ["lat", "latitude"]
        lon_dims = ["lon", "longitude"]

        def check_for_coords(coord_names_data, coord_names_expected):
            for coord in coord_names_expected:
                stat, ind_coord = check_str_in_list(coord_names_data, coord, return_ind=True)
                if stat:
                    return ind_coord[0], coord_names_data[ind_coord[0]] # just take the first value

            raise ValueError("%{0}: Could not find one of the following coordinates in the passed dictionary."
                             .format(method, ",".join(coord_names_expected)))

        lat_ind, lat_name = check_for_coords(dims, lat_dims)
        lon_ind, lon_name = check_for_coords(dims, lon_dims)

        lat, lon = np.deg2rad(scalar_field[lat_name]), np.deg2rad(scalar_field[lon_name])

        dphi, dlambda = lat[1].values - lat[0].values, lon[1].values - lon[0].values

        if order == 1:
            dvar_dlambda = 1./(r_e*np.cos(lat)*np.deg2rad(dlambda))*scalar_field.differentiate(lon_name)
            dvar_dphi = 1./(r_e*np.deg2rad(dphi))*scalar_field.differentiate(lat_name)
            dvar_dlambda = dvar_dlambda.transpose(*scalar_field.dims)    # ensure that dimension ordering is not changed

            var_diff_amplitude = np.sqrt(dvar_dlambda**2 + dvar_dphi**2)
            if avg_dom: var_diff_amplitude = var_diff_amplitude.mean(dim=[lat_name, lon_name])
        else:
            raise ValueError("%{0}: Second-order differentation is not implemenetd yet.".format(method))

        return var_diff_amplitude

