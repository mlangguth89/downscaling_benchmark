# SPDX-FileCopyrightText: 2024 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Collection of auxiliary functions for statistical evaluation and class for Score-functions
"""

__email__ = "m.langguth@fz-juelich.de"
__author__ = "Michael Langguth"
__date__ = "2024-03-08"

from typing import Union, List
try:
    from tqdm import tqdm
    l_tqdm = True
except:
    l_tqdm = False
import logging
import numpy as np
import pandas as pd
import xarray as xr
from skimage.util.shape import view_as_blocks
from handle_data_class import make_tf_dataset_allmem
from other_utils import check_str_in_list, convert_to_xarray


# basic data types
da_or_ds = Union[xr.DataArray, xr.Dataset]
list_or_str = Union[List[str], str]

# auxiliary variable for logger
logger_module_name = f"main_postprocess.{__name__}"
module_logger = logging.getLogger(logger_module_name)


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
    # get local logger
    func_logger = logging.getLogger(f"{logger_module_name}.{calculate_cond_quantiles.__name__}")

    # sanity checks
    if not isinstance(data_fcst, xr.DataArray):
        err_mess = f"data_fcst must be a DataArray, but is of type '{type(data_fcst)}'."
        func_logger.error(err_mess, stack_info=True, exc_info=True)
        raise ValueError(err_mess)

    if not isinstance(data_ref, xr.DataArray):
        err_mess = f"data_ref must be a DataArray, but is of type '{type(data_ref)}'."
        func_logger.error(err_mess, stack_info=True, exc_info=True)
        raise ValueError(err_mess)

    if not (list(data_fcst.coords) == list(data_ref.coords) and list(data_fcst.dims) == list(data_ref.dims)):
        err_mess = f"Coordinates and dimensions of data_fcst and data_ref must be the same."
        func_logger.error(err_mess, stack_info=True, exc_info=True)
        raise ValueError(err_mess)

    nquantiles = len(quantiles)
    if not nquantiles >= 3:
        err_mess = f"Quantiles must be a list/tuple of at least three float values ([0..1])."
        func_logger.error(err_mess, stack_info=True, exc_info=True)
        raise ValueError(err_mess)

    if factorization == "calibration_refinement":
        data_cond = data_fcst
        data_tar = data_ref
    elif factorization == "likelihood-base_rate":
        data_cond = data_ref
        data_tar = data_fcst
    else:
        err_mess = f"Choose either 'calibration_refinement' or 'likelihood-base_rate' for factorization"
        func_logger.error(err_mess, stack_info=True, exc_info=True)
        raise ValueError(err_mess)

    # get and set some basic attributes
    data_cond_longname = data_cond.attrs.get("longname", "conditioning_variable")
    data_cond_unit = data_cond.attrs.get("unit", "unknown")

    data_tar_longname = data_tar.attrs.get("longname", "target_variable")
    data_tar_unit = data_tar.attrs.get("unit", "unknown")

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
    
    func_logger.info(f"Start caclulating conditional quantiles for all {nbins:d} bins.")
    # fill the quantile data array
    for i in np.arange(nbins):
        # conditioning of ground truth based on forecast
        data_cropped = data_tar.where(np.logical_and(data_cond >= bins[i], data_cond < bins[i + 1]))
        # quantile-calculation
        quantile_panel.loc[dict(bin_center=bins_c[i])] = data_cropped.quantile(quantiles)

    return quantile_panel, data_cond

def get_cdf_of_x(sample_in, prob_in):
    """
    Wrappper for interpolating CDF-value for given data
    :param sample_in : input values to derive discrete CDF
    :param prob_in   : corresponding CDF
    :return: lambda function converting arbitrary input values to corresponding CDF value
    """
    return lambda xin: np.interp(xin, sample_in, prob_in)

def get_seeps_matrix(seeps_param):
    """
    Converts SEEPS paramter array to SEEPS matrix.
    :param seeps_param: Array providing p1 and p3 parameters of SEEPS weighting matrix.
    :return seeps_matrix: 3x3 weighting matrix for the SEEPS-score
    """
    # initialize matrix
    seeps_weights = xr.full_like(seeps_param["p1"], np.nan)
    seeps_weights = seeps_weights.expand_dims(dim={"weights":np.arange(9)}, axis=0).copy()
    seeps_weights.name = "SEEPS weighting matrix"
    
    # off-diagonal elements
    seeps_weights[{"weights": 1}] = 1./(1. - seeps_param["p1"])
    seeps_weights[{"weights": 2}] = 1./seeps_param["p3"] + 1./(1. - seeps_param["p1"])
    seeps_weights[{"weights": 3}] = 1./seeps_param["p1"]
    seeps_weights[{"weights": 5}] = 1./seeps_param["p3"]
    seeps_weights[{"weights": 6}] = 1./seeps_param["p1"] + 1./(1. - seeps_param["p3"])
    seeps_weights[{"weights": 7}] = 1./(1. - seeps_param["p3"])
    # diagnol elements
    seeps_weights[{"weights": [0, 4, 8]}] = xr.where(np.isnan(seeps_weights[{"weights": 7}]), np.nan, 0.)
    
    return seeps_weights

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
    # get local logger
    func_logger = logging.getLogger(f"{logger_module_name}.{perform_block_bootstrap_metric.__name__}")

    if not isinstance(metric, da_or_ds.__args__):
        err_mess = f"Input metric must be a xarray DataArray or Dataset and not {type(metric)}."
        func_logger.error(err_mess, stack_info=True, exc_info=True)
        raise ValueError(err_mess)
    
    if dim_name not in metric.dims:
        err_mess = f"Passed dimension {dim_name} cannot be found in passed metric."
        func_logger.error(err_mess, stack_info=True, exc_info=True)
        raise ValueError(err_mess)

    metric = metric.sortby(dim_name)

    dim_length = np.shape(metric.coords[dim_name].values)[0]
    nblocks = int(np.floor(dim_length/block_length))

    if nblocks < 10:
        err_mess = f"Less than 10 blocks are present with given block length {block_length:d}. Too less for bootstrapping."
        func_logger.error(err_mess, stack_info=True, exc_info=True)
        raise ValueError(err_mess)

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

    func_logger.info("Start block bootstrapping...")
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


def get_domain_info(da: xr.DataArray, lonlat_dims: list =["lon", "lat"], re:float = 6378*1.e+03): 
    """
    Get information about the underlying grid of a DataArray.
    Assumes a regular, spherical grid (can also be a rotated one of lonlat_dims are adapted accordingly)
    :param da: The xrray DataArray given on a regular, spherical grid (i.e. providing latitude and longitude coordinates)
    :param lonlat_dims: Names of the longutude and latitude coordinates
    :param re: radius of spherical Earth 
    :return grid_dict: dictionary providing dx (grid spacing), nx (#gridpoints) and Lx(domain length) (same for y) as well as lat0 (central latitude)
    """
    # get local logger
    func_logger = logging.getLogger(f"{logger_module_name}.{get_domain_info.__name__}")

    lon, lat = da[lonlat_dims[0]], da[lonlat_dims[1]]

    try:
        assert lon.ndim, f"Longitude data must be a 1D-array, but is a {lon.ndim:d}D-array"
    except AssertionError as e:
        func_logger.error(e, stack_info=True, exc_info=True)
        raise e
    
    try:
        assert lat.ndim, f"Latitude data must be a 1D-array, but is a {lat.ndim:d}D-array"
    except AssertionError as e:
        func_logger.error(e, stack_info=True, exc_info=True)
        raise e 

    lat0 = np.mean(lat)
    nx, ny = len(lon), len(lat)
    dx, dy = (lon[1] - lon[0]).values, (lat[1] - lat[0]).values 

    deg2m = re*2*np.pi/360.
    Lx, Ly = np.abs(nx*dx*deg2m)*np.cos(np.deg2rad(lat0)), np.abs(ny*dy*deg2m)

    grid_dict = {"nx": nx, "ny": ny, "dx": dx, "dy": dy, 
                 "Lx": Lx, "Ly": Ly, "lat0": lat0}
    
    return grid_dict


def detrend_data(da: xr.DataArray, xy_dims: list =["lon", "lat"]):
    """
    Detrend data on a limited area domain to majke it periodic in horizontal directions.
    Method based on Errico, 1985.
    :param da: The data given on a regular (spherical) grid
    :param xy_dims: Names of horizontal dimensions
    :return detrended, periodic data:
    """
    
    x_dim, y_dim = xy_dims[0], xy_dims[1]
    nx, ny = len(da[x_dim]), len(da[y_dim])
    
    # remove trend in x-direction
    fac_x = xr.full_like(da, 1.) * xr.DataArray(2*np.arange(nx) - nx, dims=x_dim, coords={x_dim: da[x_dim]})
    fac_y = xr.full_like(da, 1.)* xr.DataArray(2*np.arange(ny) - ny, dims=y_dim, coords={y_dim: da[y_dim]})
    trend_x, _ = xr.broadcast((da.isel({x_dim: -1}) - da.isel({x_dim: 0}))/float(nx-1), da)
    da = da - 0.5 * trend_x*fac_x
    # remove trend in y-direction
    trend_y, _ = xr.broadcast((da.isel({y_dim: -1}) - da.isel({y_dim: 0}))/float(ny-1), da)
    da = da - 0.5 * trend_y*fac_y
    
    return da


def angular_integration(da_fft, grid_dict: dict, lcutoff: bool = True):
    """
    Get power spectrum as a function of the total wavenumber.
    The integration in the (k_x, k_y)-plane is done by summation over (k_x, k_y)-pairs lying in annular rings, cf. Durran et al., 2017
    :param da_fft: Fast Fourier transformed data with (lat, lon) as last two dimensions
    :param grid_dict: dictionary providing information on underlying grid (generated by get_domain_info-method)
    :param lcutoff: flag if spectrum should be truncated to cutoff frequency or if full spectrum should be returned (False)
    :return power spectrum in total wavenumber space.
    """
    # get local logger
    func_logger = logging.getLogger(f"{logger_module_name}.{angular_integration.__name__}")

    sh = da_fft.shape
    nx, ny = grid_dict["nx"], grid_dict["ny"]
    dk = np.array([2.*np.pi/grid_dict["Lx"], 2.*np.pi/grid_dict["Ly"]])
    idkh = np.argmax(dk)
    dkx, dky = dk
    nmax = int(np.round(np.sqrt(np.square(nx*dkx) + np.square(ny*dky))/dk[idkh])) 

    sh = (*sh[:-2], nmax) if da_fft.ndim >= 2 else (1, nmax)    # add singleton dim if da_fft is a 2D-array only 
    spec_radial = np.zeros(sh, dtype="float")
    
    # start angular integration
    func_logger.info(f"Start angular integration for {nmax:d} wavenumbers.")
    for i in range(nx):
        for j in range(ny):
            k_now = int(np.round(np.sqrt(np.square(i*dkx) + np.square(j*dky))/dk[idkh]))
            spec_radial[..., k_now] += da_fft[..., i, j].real**2 + da_fft[..., i, j].imag**2

    if lcutoff:
        # Cutting/truncating the spectrum is required to ensure that the combinations of kx and ky
        # to yield the total wavenumber are complete. Without this truncation, the spectrum gets distorted 
        # as argued in Errico, 1985 (cf. Eq. 6 therein).
        # Note that Errico, 1985 chooses the maximum of (nx, ny) since dk is defined as dk=min(kx, ky).
        # Here, we choose dk = max(dkx, dky) following Durran et al., 2017 (see Eq. 18 therein), and thus have to choose min(nx, ny).
        cutoff = int(np.round(min(np.array([nx, ny]))/2 + 0.01))
        spec_radial = spec_radial[..., :cutoff]                

    return np.squeeze(spec_radial)


def get_spectrum(da: xr.DataArray, lonlat_dims = ["lon", "lat"], lcutoff: bool = True, re: float = 6378*1e+03):
    """
    Compute power spectrum in terms of total wavenumber from numpy-array.
    Note: Assumes a regular, spherical grid with dx=dy.
    :param da: DataArray with (lon, lat)-like dimensions
    :param lcutoff: flag if spectrum should be truncated to cutoff frequency or if full spectrum should be returned (False)
    :param re: (spherical) Earth radius
    :return var_rad: power spectrum in terms of wavenumber
    """
    # get local logger
    func_logger = logging.getLogger(f"{logger_module_name}.{get_spectrum.__name__}")

    # sanity check on # dimensions
    grid_dict = get_domain_info(da, lonlat_dims, re=re)
    nx, ny = grid_dict["nx"], grid_dict["ny"]
    lx, ly = grid_dict["Lx"], grid_dict["Ly"]

    da = da.transpose(..., *lonlat_dims)
    # detrend data to get periodic boundary values (cf. Errico, 1985)
    da_var = detrend_data(da, xy_dims=lonlat_dims)
    # ... and apply FFT
    func_logger.info("Start computing Fourier transformation")
    fft_var = np.fft.fft2(da_var.values)/float(nx*ny)

    var_rad = angular_integration(fft_var, {"nx": nx, "ny": ny, "Lx": lx, "Ly": ly}, lcutoff)

    return var_rad


def sample_permut_xyt(da_orig: xr.DataArray, patch_size:tuple = (6, 6)):
    """
    Permutes sample in a spatio-temporal way following the method of Breiman (2001). 
    The concrete implementation follows Höhlein et al., 2020 with spatial permutation based on patching.
    Note that the latter allows to handle time-invariant data.
    :param da_orig: original sample. Must be 3D with a 'time'-dimension 
    :param patch_size: tuple for patch size
    :return: spatio-temporally permuted sample 
    """
    # get local logger
    func_logger = logging.getLogger(f"{logger_module_name}.{sample_permut_xyt.__name__}")

    try:
        assert da_orig.ndim == 3, f"da_orig must be a 3D-array, but has {da_orig.ndim} dimensions."
    except AssertionError as e:
        func_logger.error(e, stack_info=True, exc_info=True)
        raise e

    coords_orig = da_orig.coords
    dims_orig = da_orig.dims
    sh_orig = da_orig.shape

    # temporal permutation
    func_logger.info(f"Start spatio-temporal permutation for sample with shape {sh_orig}.")

    ntimes = len(da_orig["time"])
    if dims_orig != "time":
        da_orig = da_orig.transpose("time", ...)
        coords_now, dims_now, sh_now = da_orig.coords, da_orig.dims, da_orig.shape
    else:
        coords_now, dims_now, sh_now = coords_orig, dims_orig, sh_orig

    da_permute = np.random.permutation(da_orig).copy()
    da_permute = xr.DataArray(da_permute, coords=coords_now, dims=dims_now)

    # spatial permutation with patching
    # time must be last dimension (=channel dimension)
    sh_aux = da_permute.transpose(..., "time").shape

    # Note that the order of x- and y-coordinates does not matter here
    da_patched = view_as_blocks(da_permute.transpose(..., "time").values, block_shape=(*patch_size, ntimes))

    # convert to DataArray
    sh = da_patched.shape
    dims = ["pat_x", "pat_y", "dummy", "ix", "iy", "time"]

    da_patched = xr.DataArray(da_patched, coords={dims[0]: np.arange(sh[0]), dims[1]: np.arange(sh[1]), "dummy": range(1),
                                                  dims[3]: np.arange(sh[3]), dims[4]: np.arange(sh[4]), "time": da_permute["time"]}, 
                              dims=dims)
    
    # stack xy-patches and permute
    da_patched = da_patched.stack({"pat_xy": ["pat_x", "pat_y"]})
    da_patched[...] = np.random.permutation(da_patched.transpose()).transpose()
    
    # unstack
    da_patched = da_patched.unstack().transpose(*dims)

    # revert view_as_blocks-opertaion
    da_patched = da_patched.values.transpose([0, 3, 1, 4, 2, 5]).reshape(sh_aux)
    
    # write data back on da_permute
    da_permute[...] = np.moveaxis(da_patched, 2, 0)

    # transpose to original dimension ordering if required
    da_permute = da_permute.transpose(*dims_orig)

    return da_permute


def feature_importance(ds: xr.Dataset, predictors: list_or_str, varname_tar: str, model, norm, score_name: str,
                       data_loader_opt: dict, patch_size = (6, 6)):
    """
    Run featiure importance analysis based on permutation method (see signature of sample_permut_xyt-method)
    :param ds: The unnormalized (test-)dataset
    :param predictors: List of predictor variables
    :param varname_tar: Name of target variable
    :param model: Trained model for inference
    :param norm: Normalization object
    :param score_name: Name of metric-score to be calculated
    :param data_loader_opt: Dictionary providing options for the TensorFlow data pipeline
    :param patch_size: Tuple for patch size during spatio-temporal permutation
    :return score_all: DataArray with scores for all predictor variables
    """
    # get local logger
    func_logger = logging.getLogger(f"{logger_module_name}.{feature_importance.__name__}")

    # sanity checks
    _ = check_str_in_list(list(ds.data_vars), predictors)
    #try:
    #    assert ds.dims[0] == "time", f"First dimension of the data must be a time-dimensional, but is {ds.dims[0]}."
    #except AssertionError as e:
    #    func_logger.error(e, stack_info=True, exc_info=True)
    #    raise e

    ntimes = len(ds["time"])

    # get ground truth data and underlying metadata
    ground_truth = ds[varname_tar].copy() 
    # normalize dataset
    ds = norm.normalize(ds)   

    # initialize score-array
    score_all = xr.DataArray(np.zeros((len(predictors), ntimes)), coords={"predictor": predictors, "time": ds["time"]},
                             dims=["predictor", "time"])

    for var in predictors:
        func_logger.info(f"Run sample importance analysis for {var}...")
        # get copy of sample array
        ds_copy = ds.copy(deep=True)
        # permute sample
        da_now = ds[var].copy()
        if "time" not in da_now.dims:
            da_now = da_now.expand_dims({"time": ds_copy["time"]}, axis=0)
        da_permut = sample_permut_xyt(da_now, patch_size=patch_size)
        ds_copy[var] = da_permut
        
        # get TF dataset
        func_logger.info(f"Set-up data pipeline with permuted sample for {var}...")
        tfds_test = make_tf_dataset_allmem(ds_copy, **data_loader_opt)

        # predict
        func_logger.info(f"Run inference with permuted sample for {var}...")
        y_pred = model.predict(tfds_test, verbose=2)

        # convert to xarray
        y_pred = convert_to_xarray(y_pred, norm, varname_tar, ground_truth.coords, ground_truth.dims, True)

        # calculate score
        func_logger.info(f"Calculate score for permuted samples of {var}...")
        score_engine = Scores(y_pred, ground_truth, dims=ground_truth.dims[1::])
        score_all.loc[{"predictor": var}] = score_engine(score_name)

        #free_mem([da_copy, da_permut, tfds_test, y_pred, score_engine])

    return score_all
                                    






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
            # print("Scores will be averaged across all data dimensions.")
        else:
            dim_stat = [avg_dim in self.data_dims for avg_dim in dims]
            if not all(dim_stat):
                ind_bad = [i for i, x in enumerate(dim_stat) if not x]
                raise ValueError("The following dimensions for score-averaging are not " +
                                 "part of the data: {0}".format(", ".join(np.array(dims)[ind_bad])))

            self._avg_dims = dims

    def get_2x2_event_counts(self, thresh):
        """
        Get counts of 2x2 contingency tables
        :param thres: threshold to define events
        :return: (a, b, c, d)-tuple of 2x2 contingency table
        """
        a = ((self.data_fcst >= thresh) & (self.data_ref >= thresh)).sum(dim=self.avg_dims)
        b = ((self.data_fcst >= thresh) & (self.data_ref < thresh)).sum(dim=self.avg_dims)
        c = ((self.data_fcst < thresh) & (self.data_ref >= thresh)).sum(dim=self.avg_dims)
        d = ((self.data_fcst < thresh) & (self.data_ref < thresh)).sum(dim=self.avg_dims)

        return a, b, c, d

    def calc_ets(self, thresh=0.1):
        """
        Calculates Equitable Threat Score (ETS) on data.
        :param thres: threshold to define events
        :return: ets-values
        """
        a, b, c, d = self.get_2x2_event_counts(thresh)
        n = a + b + c + d
        ar = (a + b)*(a + c)/n      # random reference forecast
        
        denom = (a + b + c - ar)

        ets = (a - ar)/denom
        ets = ets.where(denom > 0, np.nan)

        return ets
    
    def calc_fbi(self, thresh=0.1):
        """
        Calculates Frequency bias (FBI) on data.
        :param thres: threshold to define events
        :return: fbi-values
        """
        a, b, c, d = self.get_2x2_event_counts(thresh)

        denom = a+c
        fbi = (a + b)/denom

        fbi = fbi.where(denom > 0, np.nan)

        return fbi
    
    def calc_pss(self, thresh=0.1):
        """
        Calculates Peirce Skill Score (PSS) on data.
        :param thres: threshold to define events
        :return: pss-values
        """
        a, b, c, d = self.get_2x2_event_counts(thresh)      

        denom = (a + c)*(b + d)
        pss = (a*d - b*c)/denom

        pss = pss.where(denom > 0, np.nan)

        return pss   

    def calc_l1(self, **kwargs):
        """
        Calculate the L1 error norm of forecast data w.r.t. reference data.
        L1 will be divided by the number of samples along the average dimensions.
        Similar to MAE, but provides just a number divided by number of samples along average dimensions.
        :return: L1-error 
        """
        if kwargs:
            print("Passed keyword arguments to calc_l1 are without effect.")   

        l1 = np.sum(np.abs(self.data_fcst - self.data_ref))

        len_dims = np.array([self.data_fcst.sizes[dim] for dim in self.avg_dims])
        l1 /= np.prod(len_dims)

        return l1
    
    def calc_l2(self, **kwargs):
        """
        Calculate the L2 error norm of forecast data w.r.t. reference data.
        Similar to RMSE, but provides just a number divided by number of samples along average dimensions.
        :return: L2-error 
        """
        if kwargs:
            print("Passed keyword arguments to calc_l2 are without effect.")   

        l2 = np.sum(np.square(self.data_fcst - self.data_ref))

        len_dims = np.array([self.data_fcst.sizes[dim] for dim in self.avg_dims])
        l2 /= np.prod(len_dims)

        return l2
    
    def calc_mae(self, **kwargs):
        """
        Calculate mean absolute error (MAE) of forecast data w.r.t. reference data
        :return: MAE averaged over provided dimensions
        """
        if kwargs:
            print("Passed keyword arguments to calc_mae are without effect.")   

        mae = np.abs(self.data_fcst - self.data_ref).mean(dim=self.avg_dims)

        return mae

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
    
    def calc_acc(self, clim_mean: xr.DataArray, spatial_dims: List = ["lat", "lon"]):
        """
        Calculate anomaly correlation coefficient (ACC).
        :param clim_mean: climatological mean of the data
        :param spatial_dims: names of spatial dimensions over which ACC are calculated. 
                             Note: No averaging is possible over these dimensions.
        :return acc: Averaged ACC (except over spatial_dims)
        """

        fcst_ano, obs_ano = self.data_fcst - clim_mean, self.data_ref - clim_mean

        acc = (fcst_ano*obs_ano).sum(spatial_dims)/np.sqrt(fcst_ano.sum(spatial_dims)*obs_ano.sum(spatial_dims))

        mean_dims = [x for x in self.avg_dims if x not in spatial_dims]
        if len(mean_dims) > 0:
            acc = acc.mean(mean_dims)

        return acc

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
    
    def calc_iqd(self, xnodes=None, nh=0, lfilter_zero=True):
        """
        Calculates squared integrated distance between simulation and observational data.
        Method: Retrieves the empirical CDF, calculates CDF(xnodes) for both data sets and
                then uses the squared differences at xnodes for trapezodial integration.
                Note, that xnodes should be selected in a way, that CDF(xvalues) increases
                more or less continuously by ~0.01 - 0.05 for increasing xnodes-elements
                to ensure accurate integration.

        :param data_simu : 1D-array of simulation data
        :param data_obs  : 1D-array of (corresponding) observational data
        :param xnodes    : x-values used as nodes for integration (optional, automatically set if not given
                           according to stochastic properties of precipitation data)
        :param nh        : accumulation period (affects setting of xnodes)
        :param lfilter_zero: Flag to filter out zero values from CDF calculation
        :return: integrated quadrated distance between CDF of data_simu and data_obs
        """

        data_simu = self.data_fcst.values.flatten()
        data_obs = self.data_ref.values.flatten() 

        # Scarlet: because data_simu and data_obs are flattened anyway this is not needed
        #if np.ndim(data_simu) != 1 or np.ndim(data_obs) != 1:
        #    raise ValueError("Input data arrays must be 1D-arrays.")

        if xnodes is None:
            if nh == 1:
                xnodes = [0., 0.005, 0.01, 0.015, 0.025, 0.04, 0.06, 0.08, 0.1, 0.13, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6,
                          0.8, 1., 1.25, 1.5, 1.8, 2.4, 3., 3.75, 4.5, 5.25, 6., 7., 9., 12., 20., 30., 50.]
            elif 1 < nh <= 6:
                ### obtained manually based on observational data between May and July 2017
                ### except for the first step and the highest node-values,
                ### CDF is increased by 0.03 - 0.05 with every step ensuring accurate integration
                xnodes = [0.00, 0.005, 0.01, 0.02, 0.04, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.15, 1.5, 1.9, 2.4,
                         3., 4., 5., 6., 7.5, 10., 15., 25., 40., 60.]
            else:
                xnodes = [0.00, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1., 1.3, 1.7, 2.1, 2.5,
                          3., 3.5, 4., 4.75, 5.5, 6.3, 7.1, 8., 9., 10., 12.5, 15., 20., 25., 35., 50., 70., 100.]

        data_simu_filt = data_simu[~np.isnan(data_simu)]
        data_obs_filt = data_obs[~np.isnan(data_obs)]
        if lfilter_zero:
            data_simu_filt = np.sort(data_simu_filt[data_simu_filt > 0.])
            data_obs_filt  = np.sort(data_obs_filt[data_obs_filt > 0.])
        else:
            data_simu_filt = np.sort(data_simu_filt)
            data_obs_filt  = np.sort(data_obs_filt)

        nd_points_simu = np.shape(data_simu_filt)[0]
        nd_points_obs  = np.shape(data_obs_filt)[0]

        prob_simu = 1. * np.arange(nd_points_simu)/ (nd_points_simu - 1)
        prob_obs  = 1. * np.arange(nd_points_obs)/ (nd_points_obs -1)

        cdf_simu = get_cdf_of_x(data_simu_filt,prob_simu)
        cdf_obs  = get_cdf_of_x(data_obs_filt,prob_obs)

        yvals_simu = cdf_simu(xnodes)
        yvals_obs  = cdf_obs(xnodes)

        if yvals_obs[-1] < 0.999:
            print("CDF of last xnodes {0:5.2f} for observation data is smaller than 99.9%." +
                  "Consider setting xnodes manually!")

        if yvals_simu[-1] < 0.999:
            print("CDF of last xnodes {0:5.2f} for simulation data is smaller than 99.9%." +
                  "Consider setting xnodes manually!")

        # finally, perform trapezodial integration
        return np.trapz(np.square(yvals_obs - yvals_simu), xnodes)

    def calc_seeps(self, seeps_weights: xr.DataArray, t1: xr.DataArray, t3: xr.DataArray, spatial_dims: List):
        """
        Calculates stable equitable error in probabiliyt space (SEEPS), see Rodwell et al., 2011
        :param seeps_weights: SEEPS-parameter matrix to weight contingency table elements
        :param t1: threshold for light precipitation events
        :param t3: threshold for strong precipitation events
        :param spatial_dims: list/name of spatial dimensions of the data
        :return seeps skill score (i.e. 1-SEEPS)
        """

        def seeps(data_ref, data_fcst, thr_light, thr_heavy, seeps_weights):
            ob_ind = (data_ref > thr_light).astype(int) + (data_ref >= thr_heavy).astype(int)
            fc_ind = (data_fcst > thr_light).astype(int) + (data_fcst >= thr_heavy).astype(int)
            indices = fc_ind * 3 + ob_ind  # index of each data point in their local 3x3 matrices
            seeps_val = seeps_weights[indices, np.arange(len(indices))]  # pick the right weight for each data point
            
            return 1.-seeps_val
        
        if self.data_fcst.ndim == 3:
            assert len(spatial_dims) == 2, f"Provide two spatial dimensions for three-dimensional data."
            data_fcst, data_ref = self.data_fcst.stack({"xy": spatial_dims}), self.data_ref.stack({"xy": spatial_dims})
            seeps_weights = seeps_weights.stack({"xy": spatial_dims})
            t3 = t3.stack({"xy": spatial_dims})
            lstack = True
        elif self.data_fcst.ndim == 2:
            data_fcst, data_ref = self.data_fcst, self.data_ref
            lstack = False
        else:
            raise ValueError(f"Data must be a two-or-three-dimensional array.")

        # check dimensioning of data
        assert data_fcst.ndim <= 2, f"Data must be one- or two-dimensional, but has {data_fcst.ndim} dimensions. Check if stacking with spatial_dims may help." 

        if data_fcst.ndim == 1:
            seeps_values_all = seeps(data_ref, data_fcst, t1.values, t3, seeps_weights)
        else:
            data_fcst, data_ref = data_fcst.transpose(..., "xy"), data_ref.transpose(..., "xy")
            seeps_values_all = xr.full_like(data_fcst, np.nan)
            seeps_values_all.name = "seeps"
            for it in range(data_ref.shape[0]):
                data_fcst_now, data_ref_now = data_fcst[it, ...], data_ref[it, ...]
                # in case of missing data, skip computation
                if np.all(np.isnan(data_fcst_now)) or np.all(np.isnan(data_ref_now)):
                    continue

                seeps_values_all[it,...] = seeps(data_ref_now, data_fcst_now, t1.values, t3, seeps_weights.values)

        if lstack:
            seeps_values_all = seeps_values_all.unstack()

        seeps_values = seeps_values_all.mean(dim=self.avg_dims)

        return seeps_values

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
