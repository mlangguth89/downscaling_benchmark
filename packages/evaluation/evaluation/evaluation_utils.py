# SPDX-FileCopyrightText: 2025 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC); Gesosphere Austria (GSA)
#
# SPDX-License-Identifier: MIT

"""
Collection of statistical evaluation methods used in postprocess.py.
"""

__email__ = "m.langguth@fz-juelich.de"
__author__ = "Michael Langguth"
__date__ = "2024-04-19"
__updated__ = "2025-05-16"

import logging
import os
import sys
from typing import Any, List, Union

import numpy as np
import xarray as xr
from skimage.util.shape import view_as_blocks

str_or_List = Union[List, str]

# basic data types
da_or_ds = Union[xr.DataArray, xr.Dataset]
list_or_str = Union[List[str], str]

# auxiliary variable for logger
logger_module_name = f"main_postprocess.{__name__}"
module_logger = logging.getLogger(logger_module_name)

def calculate_cond_quantiles(
    data_fcst: xr.DataArray,
    data_ref: xr.DataArray,
    data_names: List[str],
    unit: str,
    factorization="calibration_refinement",
    quantiles=(0.05, 0.5, 0.95),
):
    """
    Calculate conditional quantiles of forecast and observation/reference data with selected factorization
    :param data_fcst: forecast data array
    :param data_ref: observational/reference data array
    :param factorization: factorization: "likelihood-base_rate" p(m|o) or "calibration_refinement" p(o|m)-> default
    :param quantiles: conditional quantiles
    :return quantile_panel: conditional quantiles of p(m|o) or p(o|m)
    """
    # get local logger
    func_logger = logging.getLogger(
        f"{logger_module_name}.{calculate_cond_quantiles.__name__}"
    )

    # sanity checks
    if not isinstance(data_fcst, xr.DataArray):
        err_mess = f"data_fcst must be a DataArray, but is of type '{type(data_fcst)}'."
        func_logger.error(err_mess, stack_info=True, exc_info=True)
        raise ValueError(err_mess)

    if not isinstance(data_ref, xr.DataArray):
        err_mess = f"data_ref must be a DataArray, but is of type '{type(data_ref)}'."
        func_logger.error(err_mess, stack_info=True, exc_info=True)
        raise ValueError(err_mess)

    if not (
        list(data_fcst.coords) == list(data_ref.coords)
        and list(data_fcst.dims) == list(data_ref.dims)
    ):
        err_mess = (
            f"Coordinates and dimensions of data_fcst and data_ref must be the same."
        )
        func_logger.error(err_mess, stack_info=True, exc_info=True)
        raise ValueError(err_mess)

    nquantiles = len(quantiles)
    if not nquantiles >= 3:
        err_mess = (
            f"Quantiles must be a list/tuple of at least three float values ([0..1])."
        )
        func_logger.error(err_mess, stack_info=True, exc_info=True)
        raise ValueError(err_mess)

    if factorization == "calibration_refinement":
        data_cond, data_tar = data_fcst, data_ref
        data_cond_name, data_tar_name = data_names[0], data_names[1]
    elif factorization == "likelihood-base_rate":
        data_cond, data_tar = data_ref, data_fcst
        data_cond_name, data_tar_name = data_names[1], data_names[0]
    else:
        err_mess = f"Choose either 'calibration_refinement' or 'likelihood-base_rate' for factorization. Your choice {factorization}"
        func_logger.error(err_mess, stack_info=True, exc_info=True)
        raise ValueError(err_mess)

    # get bins for conditioning
    data_cond_min, data_cond_max = (
        np.floor(np.min(data_cond)),
        np.ceil(np.max(data_cond)),
    )
    bins = list(np.arange(int(data_cond_min), int(data_cond_max) + 1))
    bins_c = 0.5 * (np.asarray(bins[0:-1]) + np.asarray(bins[1:]))
    nbins = len(bins) - 1

    # get all possible bins from target and conditioning variable
    data_all_min, data_all_max = (
        np.minimum(data_cond_min, np.floor(np.min(data_tar))),
        np.maximum(data_cond_max, np.ceil(np.max(data_tar))),
    )
    bins_all = list(np.arange(int(data_all_min), int(data_all_max) + 1))
    bins_c_all = 0.5 * (np.asarray(bins_all[0:-1]) + np.asarray(bins_all[1:]))
    # initialize quantile data array
    quantile_panel = xr.DataArray(
        np.full((len(bins_c_all), nquantiles), np.nan),
        coords={"bin_center": bins_c_all, "quantile": list(quantiles)},
        dims=["bin_center", "quantile"],
        attrs={
            "cond_varname": data_cond_name,
            "unit": unit,
            "tar_varname": data_tar_name,
        },
    )

    func_logger.info(f"Start caclulating conditional quantiles for all {nbins:d} bins.")
    # fill the quantile data array
    for i in np.arange(nbins):
        # conditioning of ground truth based on forecast
        data_cropped = data_tar.where(
            np.logical_and(data_cond >= bins[i], data_cond < bins[i + 1])
        )
        # quantile-calculation
        quantile_panel.loc[dict(bin_center=bins_c[i])] = data_cropped.quantile(
            quantiles
        )

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
    seeps_weights = seeps_weights.expand_dims(
        dim={"weights": np.arange(9)}, axis=0
    ).copy()
    seeps_weights.name = "SEEPS weighting matrix"

    # off-diagonal elements
    seeps_weights[{"weights": 1}] = 1.0 / (1.0 - seeps_param["p1"])
    seeps_weights[{"weights": 2}] = 1.0 / seeps_param["p3"] + 1.0 / (
        1.0 - seeps_param["p1"]
    )
    seeps_weights[{"weights": 3}] = 1.0 / seeps_param["p1"]
    seeps_weights[{"weights": 5}] = 1.0 / seeps_param["p3"]
    seeps_weights[{"weights": 6}] = 1.0 / seeps_param["p1"] + 1.0 / (
        1.0 - seeps_param["p3"]
    )
    seeps_weights[{"weights": 7}] = 1.0 / (1.0 - seeps_param["p3"])
    # diagnol elements
    seeps_weights[{"weights": [0, 4, 8]}] = xr.where(
        np.isnan(seeps_weights[{"weights": 7}]), np.nan, 0.0
    )

    return seeps_weights


def perform_block_bootstrap_metric(
    metric: da_or_ds,
    dim_name: str,
    block_length: int,
    nboots_block: int = 1000,
    seed: int = 42,
):
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
    func_logger = logging.getLogger(
        f"{logger_module_name}.{perform_block_bootstrap_metric.__name__}"
    )

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
    nblocks = int(np.floor(dim_length / block_length))

    if nblocks < 10:
        err_mess = f"Less than 10 blocks are present with given block length {block_length:d}. Too less for bootstrapping."
        func_logger.warning(err_mess, stack_info=False, exc_info=False)
        return None
        # raise ValueError(err_mess)

    # precompute metrics of block
    for iblock in np.arange(nblocks):
        ind_s, ind_e = iblock * block_length, (iblock + 1) * block_length
        metric_block_aux = metric.isel({dim_name: slice(ind_s, ind_e)}).mean(
            dim=dim_name
        )
        if iblock == 0:
            metric_val_block = metric_block_aux.expand_dims(
                dim={"iblock": 1}, axis=0
            ).copy(deep=True)
        else:
            metric_val_block = xr.concat(
                [
                    metric_val_block,
                    metric_block_aux.expand_dims(dim={"iblock": 1}, axis=0),
                ],
                dim="iblock",
            )

    metric_val_block["iblock"] = np.arange(nblocks)

    # get random blocks
    np.random.seed(seed)
    iblocks_boot = np.sort(np.random.randint(nblocks, size=(nboots_block, nblocks)))

    for iboot_b in range(nboots_block):
        metric_boot_aux = metric_val_block.isel(iblock=iblocks_boot[iboot_b, :]).mean(
            dim="iblock"
        )
        if iboot_b == 0:
            metric_boot = metric_boot_aux.expand_dims(dim={"iboot": 1}, axis=0).copy(
                deep=True
            )
        else:
            metric_boot = xr.concat(
                [metric_boot, metric_boot_aux.expand_dims(dim={"iboot": 1}, axis=0)],
                dim="iboot",
            )

    # set iboot-coordinate
    metric_boot["iboot"] = np.arange(nboots_block)
    if isinstance(metric_boot, xr.Dataset):
        new_varnames = ["{0}_bootstrapped".format(var) for var in metric.data_vars]
        metric_boot = metric_boot.rename(dict(zip(metric.data_vars, new_varnames)))

    return metric_boot


def bootstrap_grouped_hourly(
    score_hourly_grouped, score_hourly_mean, block_length, nboots
):
    """
    Perform block bootstrapping on grouped hourly scores
    :param score_hourly_grouped: Grouped hourly scores
    :param score_hourly_mean: Mean scores array
    :param block_length: Block length for bootstrapping
    :param nboots: Number of bootstraps
    """
    # get local logger
    func_logger = logging.getLogger(
        f"{logger_module_name}.{bootstrap_grouped_hourly.__name__}"
    )

    # initialize arrays for block bootstrapping
    dims_boot = to_list(score_hourly_mean.dims) + ["iboot"]
    coords_boot = dict(score_hourly_mean.coords).copy()
    coords_boot["iboot"] = np.arange(nboots)

    score_hourly_mean_b = xr.DataArray(
        np.zeros((len(score_hourly_mean), nboots)), dims=dims_boot, coords=coords_boot
    )

    # run block bootstrapping for group
    for hh, score_hh in score_hourly_grouped:
        func_logger.debug(f"Run block bootstrapping at {hh:02d} UTC...")
        score_hourly_mean_b.sel({"hour": hh})[...] = perform_block_bootstrap_metric(
            score_hh, "time", block_length, nboots
        )

    return score_hourly_mean_b


def get_spectrum_exps(
    ds: xr.Dataset, data_vars: List[str], data_info: dict, lcutoff: bool, re: float
):
    """
    Small wrapper to run spectral analysis for several experiments
    :param ds: xarray.Dataset with input data
    :param data_vars: List of variable names from ds for spectral analysis
    :param data_info: Dictionary with information about data, required keys: "lonlat_dims", "dims", "coord_dict", "varname", "var_unit"
    :param lcutoff: Flag to apply low-pass filter
    :param re: Earth radius
    :return: xarray.Dataset with power spectra for each variable
    """
    # get local logger
    func_logger = logging.getLogger(
        f"{logger_module_name}.{get_spectrum_exps.__name__}"
    )

    # initialize dictionary for power spectra
    ps_dict = {}

    # get information about data for convenience
    nspecs = len(data_vars)
    lonlat_dims, dims = data_info["lonlat_dims"], data_info["dims"]
    coord_dict = data_info["coord_dict"]
    var_name, var_unit = data_info["varname"], data_info["var_unit"]

    for i, data_var in enumerate(data_vars):
        func_logger.info(
            f"Start spectral analysis for experiment {data_var} ({i + 1}/{nspecs})..."
        )

        # run spectral analysis
        ps_exp = get_spectrum(
            ds[data_var], lonlat_dims=lonlat_dims, lcutoff=lcutoff, re=re
        )
        # average over all time steps and create xarray.DataArray
        da_ps_exp = xr.DataArray(
            ps_exp.mean(axis=0),
            dims=dims,
            coords=coord_dict,
            name=f"sp_{data_var}",
            attrs={
                "long_name": f"Spectral power of {data_var}",
                "units": var_unit,
                "physical variable name": var_name,
            },
        )

        # remove wavenumber 0 and append dictionary
        ps_dict[data_var] = da_ps_exp[1::]

    # turn dictionary into xarray.Dataset
    ds_ps = xr.Dataset(ps_dict)

    return ds_ps


def get_domain_info(
    da: xr.DataArray, lonlat_dims: list = ["lon", "lat"], re: float = 6378 * 1.0e03
):
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
        assert lon.ndim, (
            f"Longitude data must be a 1D-array, but is a {lon.ndim:d}D-array"
        )
    except AssertionError as e:
        func_logger.error(e, stack_info=True, exc_info=True)
        raise e

    try:
        assert lat.ndim, (
            f"Latitude data must be a 1D-array, but is a {lat.ndim:d}D-array"
        )
    except AssertionError as e:
        func_logger.error(e, stack_info=True, exc_info=True)
        raise e

    lat0 = np.mean(lat)
    nx, ny = len(lon), len(lat)
    dx, dy = (lon[1] - lon[0]).values, (lat[1] - lat[0]).values

    deg2m = re * 2 * np.pi / 360.0
    Lx, Ly = np.abs(nx * dx * deg2m) * np.cos(np.deg2rad(lat0)), np.abs(ny * dy * deg2m)

    grid_dict = {
        "nx": nx,
        "ny": ny,
        "dx": dx,
        "dy": dy,
        "Lx": Lx,
        "Ly": Ly,
        "lat0": lat0,
    }

    return grid_dict


def detrend_data(da: xr.DataArray, xy_dims: list = ["lon", "lat"]):
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
    fac_x = xr.full_like(da, 1.0) * xr.DataArray(
        2 * np.arange(nx) - nx, dims=x_dim, coords={x_dim: da[x_dim]}
    )
    fac_y = xr.full_like(da, 1.0) * xr.DataArray(
        2 * np.arange(ny) - ny, dims=y_dim, coords={y_dim: da[y_dim]}
    )
    trend_x, _ = xr.broadcast(
        (da.isel({x_dim: -1}) - da.isel({x_dim: 0})) / float(nx - 1), da
    )
    da = da - 0.5 * trend_x * fac_x
    # remove trend in y-direction
    trend_y, _ = xr.broadcast(
        (da.isel({y_dim: -1}) - da.isel({y_dim: 0})) / float(ny - 1), da
    )
    da = da - 0.5 * trend_y * fac_y

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
    func_logger = logging.getLogger(
        f"{logger_module_name}.{angular_integration.__name__}"
    )

    sh = da_fft.shape
    nx, ny = grid_dict["nx"], grid_dict["ny"]
    dk = np.array([2.0 * np.pi / grid_dict["Lx"], 2.0 * np.pi / grid_dict["Ly"]])
    idkh = np.argmax(dk)
    dkx, dky = dk
    nmax = int(np.round(np.sqrt(np.square(nx * dkx) + np.square(ny * dky)) / dk[idkh]))

    sh = (
        (*sh[:-2], nmax) if da_fft.ndim >= 2 else (1, nmax)
    )  # add singleton dim if da_fft is a 2D-array only
    spec_radial = np.zeros(sh, dtype="float")

    # start angular integration
    func_logger.info(f"Start angular integration for {nmax:d} wavenumbers.")
    for i in range(nx):
        for j in range(ny):
            k_now = int(
                np.round(np.sqrt(np.square(i * dkx) + np.square(j * dky)) / dk[idkh])
            )
            spec_radial[..., k_now] += (
                da_fft[..., i, j].real ** 2 + da_fft[..., i, j].imag ** 2
            )

    if lcutoff:
        # Cutting/truncating the spectrum is required to ensure that the combinations of kx and ky
        # to yield the total wavenumber are complete. Without this truncation, the spectrum gets distorted
        # as argued in Errico, 1985 (cf. Eq. 6 therein).
        # Note that Errico, 1985 chooses the maximum of (nx, ny) since dk is defined as dk=min(kx, ky).
        # Here, we choose dk = max(dkx, dky) following Durran et al., 2017 (see Eq. 18 therein), and thus have to choose min(nx, ny).
        cutoff = int(np.round(min(np.array([nx, ny])) / 2 + 0.01))
        spec_radial = spec_radial[..., :cutoff]

    return np.squeeze(spec_radial)


def get_spectrum(
    da: xr.DataArray,
    lonlat_dims=["lon", "lat"],
    lcutoff: bool = True,
    re: float = 6378 * 1e03,
):
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
    fft_var = np.fft.fft2(da_var.values) / float(nx * ny)

    var_rad = angular_integration(
        fft_var, {"nx": nx, "ny": ny, "Lx": lx, "Ly": ly}, lcutoff
    )

    return var_rad


def sample_permut_xyt(da_orig: xr.DataArray, patch_size: tuple = (8, 8)):
    """
    Permutes sample in a spatio-temporal way following the method of Breiman (2001).
    The concrete implementation follows Höhlein et al., 2020 with spatial permutation based on patching.
    Note that the latter allows to handle time-invariant data.
    :param da_orig: original sample. Must be 3D with a 'time'-dimension
    :param patch_size: tuple for patch size
    :return: spatio-temporally permuted sample
    """
    # get local logger
    func_logger = logging.getLogger(
        f"{logger_module_name}.{sample_permut_xyt.__name__}"
    )

    try:
        assert da_orig.ndim == 3, (
            f"da_orig must be a 3D-array, but has {da_orig.ndim} dimensions."
        )
    except AssertionError as e:
        func_logger.error(e, stack_info=True, exc_info=True)
        raise e

    coords_orig = da_orig.coords
    dims_orig = da_orig.dims
    sh_orig = da_orig.shape

    # temporal permutation
    func_logger.info(
        f"Start spatio-temporal permutation for sample with shape {sh_orig}."
    )

    ntimes = len(da_orig["time"])
    if dims_orig[0] != "time":
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
    da_patched = view_as_blocks(
        da_permute.transpose(..., "time").values, block_shape=(*patch_size, ntimes)
    )

    # convert to DataArray
    sh = da_patched.shape
    dims = ["pat_x", "pat_y", "dummy", "ix", "iy", "time"]

    da_patched = xr.DataArray(
        da_patched,
        coords={
            dims[0]: np.arange(sh[0]),
            dims[1]: np.arange(sh[1]),
            "dummy": range(1),
            dims[3]: np.arange(sh[3]),
            dims[4]: np.arange(sh[4]),
            "time": da_permute["time"],
        },
        dims=dims,
    )

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


def merge_dicts(default_dict, user_dict, recursive=True):
    """
    Merge two dictionaries, ensuring that all default keys are set.
    Recursive strategy is optional.
    :param default_dict: Dictionary with default values.
    :param user_dict: Dictionary with user-specified values.
    :param recursive: If True, recursively merge nested dictionaries.
    :return: Merged dictionary.
    """
    merged_dict = default_dict.copy()

    for key, value in user_dict.items():
        if key not in merged_dict:
            raise KeyError(f"Key '{key}' not found in the default dictionary.")

        default_value = merged_dict.get(key)

        if isinstance(value, dict) and isinstance(default_value, dict) and recursive:
            merged_dict[key] = merge_dicts(default_value, value)
        else:
            if default_value is not None:
                expected_type = type(default_value)

                # Convert lists/tuples to match the expected type
                if isinstance(value, (list, tuple)) and isinstance(
                    default_value, (list, tuple)
                ):
                    value = (
                        tuple(value)
                        if isinstance(default_value, tuple)
                        else list(value)
                    )

                assert isinstance(value, expected_type), (
                    f"Type mismatch for key '{key}': Expected {expected_type}, got {type(value)}"
                )

            merged_dict[key] = value

    return merged_dict


def to_list(obj: Any) -> List:
    """
    Method from MLAIR!
    Transform given object to list if obj is not already a list. Sets are also transformed to a list
    :param obj: object to transform to list
    :return: list containing obj, or obj itself (if obj was already a list)
    """
    if isinstance(obj, (set, tuple)):
        obj = list(obj)
    elif not isinstance(obj, list):
        obj = [obj]
    return obj


def convert_to_xarray(mout_np, norm, varname, coords, dims, z_branch=False):
    """
    Converts numpy-array of model output to xarray.DataArray and performs denormalization.
    :param mout_np: numpy-array of model output
    :param norm: normalization object
    :param varname: name of variable
    :param coords: coordinates of target data
    :param dims: dimensions of target data
    :param z_branch: flag for z-branch
    :return: xarray.DataArray of model output with denormalized data
    """
    if z_branch:
        # slice data to get first channel only
        if isinstance(mout_np, list):
            mout_np = mout_np[0]
        mout_xr = xr.DataArray(
            mout_np[..., 0].squeeze(), coords=coords, dims=dims, name=varname
        )
    else:
        # no slicing required
        mout_xr = xr.DataArray(
            mout_np.squeeze(), coords=coords, dims=dims, name=varname
        )

    # get variable-specific normalizer and perform denormalization
    norm_var = norm.get_normalizer_for_var(varname)

    mout_xr = norm_var.denormalize(mout_xr, varname=varname)

    return mout_xr


def check_str_in_list(
    list_in: List, str2check: str_or_List, labort: bool = True, return_ind: bool = False
):
    """
    Checks if all strings are found in list
    :param list_in: input list
    :param str2check: string or list of strings to be checked if they are part of list_in
    :param labort: Flag if error will be risen in case of missing string in list
    :param return_ind: Flag if index for each string found in list will be returned
    :return: True if existence of all strings was confirmed, if return_ind is True, the index of each string in list is
             returned as well
    """
    stat = False
    if isinstance(str2check, str):
        str2check = [str2check]
    elif isinstance(str2check, list):
        assert np.all([isinstance(str1, str) for str1 in str2check]), (
            "Not all elements of str2check are strings"
        )
    else:
        raise ValueError(
            "str2check-argument must be either a string or a list of strings"
        )

    stat_element = [True if str1 in list_in else False for str1 in str2check]

    if np.all(stat_element):
        stat = True
    else:
        print("The following elements are not part of the input list:")
        inds_miss = np.where(list(~np.array(stat_element)))[0]
        for i in inds_miss:
            print("* index {0:d}: {1}".format(i, str2check[i]))
        if labort:
            raise ValueError("Could not find all expected strings in list.")
    # return
    if stat and not return_ind:
        return stat
    elif stat:
        return stat, [list_in.index(str_curr) for str_curr in str2check]
    else:
        return stat, []


def finditem(d, key, default=None):
    """
    Return a value corresponding to the specified key in the (possibly
    nested) dictionary d. If there is no item with that key, return
    default.
    :param d: the potentially nested dictionary
    :param key: the key to find
    :param default: default value for the key
    """
    stack = [iter(d.items())]
    while stack:
        for k, v in stack[-1]:
            if isinstance(v, dict):
                stack.append(iter(v.items()))
                break
            elif k == key:
                return v
        else:
            stack.pop()
    if default is not None:
        return default
    else:
        raise KeyError(f"Key {key} has not been found in dictionary")


def config_logger(
    logger,
    logfile: str,
    log_level_file=logging.DEBUG,
    log_level_console=logging.INFO,
    remove_existing_file: bool = True,
):

    if remove_existing_file and os.path.isfile(logfile):
        os.remove(logfile)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s: %(message)s")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')

    fh = logging.FileHandler(logfile)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level_console)
    fh.setLevel(log_level_file)

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh), logger.addHandler(ch)

    return logger
