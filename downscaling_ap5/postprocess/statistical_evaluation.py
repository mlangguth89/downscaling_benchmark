# SPDX-FileCopyrightText: 2021 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Collection of auxiliary functions for statistical evaluation and class for Score-functions
"""

__email__ = "m.langguth@fz-juelich.de"
__author__ = "Michael Langguth"
__date__ = "2022-09-11"

import numpy as np
import xarray as xr
from typing import Union, List
import os, sys, glob
sys.path.append('../')
try:
    from tqdm import tqdm
    l_tqdm = True
except:
    l_tqdm = False
from downscaling_ap5.utils.other_utils import provide_default, check_str_in_list

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

    def __init__(self, data_fcst: xr.DataArray, data_ref: xr.DataArray, dims: List[str]):
        """
        :param data_fcst: forecast data to evaluate
        :param data_ref: reference or ground truth data
        """
        self.metrics_dict = {"mse": self.calc_mse, "rmse": self.calc_rmse, "bias": self.calc_bias,
                             "grad_amplitude": self.calc_spatial_variability, "psnr": self.calc_psnr}
        self.data_fcst = data_fcst
        self.data_dims = list(self.data_fcst.dims)
        self.data_ref = data_ref
        self.avg_dims = dims

    def __call__(self, score_name, **kwargs):
        try:
            score_func = self.metrics_dict[score_name]
        except:
            raise ValueError(f"{score_name} is not an implemented score." +
                             "Choose one of the following: {0}".format(", ".join(self.metrics_dict.keys())))

        return score_func(**kwargs)

    @property
    def data_fcst(self):
        return self._data_fcst

    @data_fcst.setter
    def data_fcst(self, da_fcst):
        if not isinstance(da_fcst, xr.DataArray):
            raise ValueError("data_fcst must be a xarray DataArray.")

        self._data_fcst = da_fcst

    @property
    def data_ref(self):
        return self._data_ref

    @data_ref.setter
    def data_ref(self, da_ref):
        if not isinstance(da_ref, xr.DataArray):
            raise ValueError("data_fcst must be a xarray DataArray.")

        if not list(da_ref.dims) == self.data_dims:
            raise ValueError("Dimensions of data_fcst and data_ref must match, but got:" +
                             "[{0}] vs. [{1}]".format(", ".join(list(da_ref.dims)),
                                                      ", ".join(self.data_dims)))

        self._data_ref = da_ref

    @property
    def avg_dims(self):
        return self._avg_dims

    @avg_dims.setter
    def avg_dims(self, dims):
        if dims is None:
            self.avg_dims = self.data_dims
            print("Scores will be averaged across all data dimensions.")
        else:
            dim_stat = [avg_dim in self.data_dims for avg_dim in dims]
            if not all(dim_stat):
                ind_bad = [i for i, x in enumerate(dim_stat) if x]
                raise ValueError("The following dimensions for score-averaging are not" +
                                 "part of the data: {0}".format(", ".join(dims[ind_bad])))

            self._avg_dims = dims

    def calc_mse(self, **kwargs):
        """
        Calculate mse of forecast data w.r.t. reference data
        :return: averaged mse for each batch example, [batch,fore_hours]
        """
        if kwargs:
            print("Passed keyword arguments to calc_mse are without effect.")

        mse = np.square(self.data_fcst - self.data_ref).mean(dim=self.avg_dims)

        return mse

    def calc_rmse(self, **kwargs):

        rmse = np.sqrt(self.calc_mse(**kwargs))

        return rmse

    def calc_bias(self, **kwargs):

        if kwargs:
            print("Passed keyword arguments to calc_bias are without effect.")

        bias = (self.data_fcst - self.data_ref).mean(dim=self.avg_dims)

        return bias

    def calc_psnr(self, **kwargs):
        """
        Calculate PSNR of forecast data w.r.t. reference data
        :param kwargs: known keyword argument 'pixel_max' for maximum value of data
        :return: averaged PSNR
        """
        pixel_max = kwargs.get("pixel_max", 1.)

        mse = self.calc_mse()
        if np.count_nonzero(mse) == 0:
            psnr = mse
            psnr[...] = 100.
        else:
            psnr = 20. * np.log10(pixel_max / np.sqrt(mse))

        return psnr

    def calc_spatial_variability(self, **kwargs):
        """
        Calculates the ratio between the spatial variability of differental operator with order 1 (or 2) forecast and
        reference data using the calc_geo_spatial-method.
        :param kwargs: 'order' to control the order of spatial differential operator
                       'non_spatial_avg_dims' to add averaging in addition to spatial averaging performed with calc_geo_spatial
        :return: the ratio between spatial variabilty in the forecast and reference data field
        """
        order = kwargs.get("order", 1)
        avg_dims = kwargs.get("non_spatial_avg_dims", None)

        fcst_grad = self.calc_geo_spatial_diff(self.data_fcst, order=order)
        ref_grd = self.calc_geo_spatial_diff(self.data_ref, order=order)

        ratio_spat_variability = (fcst_grad / ref_grd)
        if avg_dims is not None:
            ratio_spat_variability = ratio_spat_variability.mean(dim=avg_dims)

        return ratio_spat_variability

    @staticmethod
    def calc_geo_spatial_diff(scalar_field: xr.DataArray, order: int = 1, r_e: float = 6371.e3, dom_avg: bool = True):
        """
        Calculates the amplitude of the gradient (order=1) or the Laplacian (order=2) of a scalar field given on a regular,
        geographical grid (i.e. dlambda = const. and dphi=const.)
        :param scalar_field: scalar field as data array with latitude and longitude as coordinates
        :param order: order of spatial differential operator
        :param r_e: radius of the sphere
        :return: the amplitude of the gradient/laplacian at each grid point or over the whole domain (see avg_dom)
        """
        method = Scores.calc_geo_spatial_diff.__name__
        # sanity checks
        assert isinstance(scalar_field, xr.DataArray), f"Scalar_field of {method} must be a xarray DataArray."
        assert order in [1, 2], f"Order for {method} must be either 1 or 2."

        dims = list(scalar_field.dims)
        lat_dims = ["rlat", "lat", "latitude"]
        lon_dims = ["rlon", "lon", "longitude"]

        def check_for_coords(coord_names_data, coord_names_expected):
            stat = False
            for i, coord in enumerate(coord_names_expected):
                if coord in coord_names_data:
                    stat = True
                    break

            if stat:
                return i, coord_names_expected[i]  # just take the first value
            else:
                raise ValueError("Could not find one of the following coordinates in the passed dictionary: {0}"
                                 .format(",".join(coord_names_expected)))

        lat_ind, lat_name = check_for_coords(dims, lat_dims)
        lon_ind, lon_name = check_for_coords(dims, lon_dims)

        lat, lon = np.deg2rad(scalar_field[lat_name]), np.deg2rad(scalar_field[lon_name])
        dphi, dlambda = lat[1].values - lat[0].values, lon[1].values - lon[0].values

        if order == 1:
            dvar_dlambda = 1. / (r_e * np.cos(lat) * dlambda) * scalar_field.differentiate(lon_name)
            dvar_dphi = 1. / (r_e * dphi) * scalar_field.differentiate(lat_name)
            dvar_dlambda = dvar_dlambda.transpose(*scalar_field.dims)  # ensure that dimension ordering is not changed

            var_diff_amplitude = np.sqrt(dvar_dlambda ** 2 + dvar_dphi ** 2)
            if dom_avg: var_diff_amplitude = var_diff_amplitude.mean(dim=[lat_name, lon_name])
        else:
            raise ValueError(f"Second-order differentation is not implemenetd in {method} yet.")

        return var_diff_amplitude
