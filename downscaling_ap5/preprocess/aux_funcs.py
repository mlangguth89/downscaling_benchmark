from collections import OrderedDict
import datetime as dt
import glob
import logging
import os
import subprocess as sp
from typing import List, Tuple

import numpy as np
import xarray as xr

from abstract_preprocess import CDOGridDes
from other_utils import flatten, remove_files, to_list
from tools_utils import CDO, NCAP2, NCEA, NCKS, NCRENAME


# ---------------------Abstract--------------------------------------------------

# merge netCDF for intersection of datums => ? erlaube werte zu überschreiben
def merge_two_netcdf(nc1: str, nc2: str, nc_tar: str, merge_dim: str = "time"):
    """
    Merge datasets from two netCDF-files. Different than cdo's merge- or mergetime-operator, the datums in both
    datasets must not coincide, but can overlap. The data will then be merged for the intersection of both datums.
    :param nc1: path to first netCDF-file to merge; dataset must include dimension merge_dim
    :param nc2: path to second netCDF-file to merge; dataset must include dimension merge_dim
    :param merge_dim: name of dimension along which datsets will be merged
    :param nc_tar: path to netCDf-file of merged dataset
    :return stat: status if merging was successful
    """
    ds1, ds2 = xr.open_dataset(nc1), xr.open_dataset(nc2)

    times1, times2 = list(ds1[merge_dim].values), list(ds2[merge_dim].values)
    joint_times = sorted(list(set(times1) & set(times2)))

    stat = True
    # try:
    if not joint_times:
        raise ValueError(
            f"No intersection on dimension {merge_dim} found for datasets."
        )
    ds_merged = xr.merge(
        [ds1.sel({merge_dim: joint_times}), ds2.sel({merge_dim: joint_times})]
    )
    ds_merged.to_netcdf(nc_tar)
    # except:
    #    stat = False

    return stat

# --------------------------abstract (cdo-griddes)---------------------------------

# validation => will be handeled by pydantic
def check_gdes_dict(grid_des_dict, lbreak=False):
    """
    Check if grid description dictionary only contains valid keys.
    :param grid_des_dict: the grid description dictionary to be checked.
    :param lbreak: Flag if error should be raised. Alternatively, a warning is printed.
    :return: -
    """

    assert isinstance(grid_des_dict, dict), "%Grid description must be a dictionary."

    gdes_keys = list(grid_des_dict.keys())
    gdes_keys_stat = [key_req in CDOGridDes.valid_keys for key_req in gdes_keys]

    if not all(gdes_keys_stat):
        invalid_keys = [
            gdes_keys[i] for i in range(len(gdes_keys)) if not gdes_keys_stat[i]
        ]
        err_str = f"%: The following keys are not valid: {', '.join(invalid_keys)}"
        if lbreak:
            raise ValueError(err_str)
        else:
            print("WARNING: " + err_str)

# construct CDOgrid from file => handle in CDOGrid class
def read_grid_des(grid_des_file):
    """
    Read CDO grid description file and put data into dictionary.
    :param grid_des_file: the grid description file to be read
    :return: dictionary with key-values from grid description parameters
                (e.g. gridtype = lonlat -> {"gridtype": "lonlat"}).
    """
    method = CDOGridDes.read_grid_des.__name__

    if not os.path.isfile(grid_des_file):
        raise FileNotFoundError(
            "%{0}: Cannot find grid description file '{1}'.".format(
                method, grid_des_file
            )
        )

    # read the file ...
    with open(grid_des_file, "r") as fgdes:
        lines = fgdes.readlines()

    # and put data into dictionary
    grid_des_dict = {}
    for line in lines:
        splitted = line.replace("\n", "").split("=")
        if len(splitted) == 2:
            grid_des_dict[splitted[0].strip()] = splitted[1].strip()

    if not grid_des_file:
        raise ValueError(
            "%{0}: Dictionary from grid description file '{1}' is empty. Please check input.".format(
                method, grid_des_file
            )
        )
    else:
        grid_des_dict["file"] = grid_des_file

    return grid_des_dict

# use other_dict.copy(), other_dict.update(first_dict)
def merge_dicts(first_dict, other_dict, create_copy: bool = False):
    """
    Merges two dicts. Keys that reside in both dictionaries are taken from the first dictionary.
    :param first_dict: first dictionary to be merged
    :param other_dict: second dictionary to be merged
    :param create_copy: Creates a new copy if True. If False, changes other_dict in-place!
    :return: merged dictionary
    """
    if create_copy:
        new_dict = other_dict.copy()
    else:
        new_dict = other_dict

    for key in first_dict.keys():
        new_dict[key] = first_dict[key]

    return new_dict

# get opposite corner => ?
def get_slice_coords(coord0, dx, n, d=4):
    """
    Small helper to get coords for slicing
    """
    coord0 = np.float(coord0)
    coords = (
        np.round(coord0, decimals=d),
        np.round(coord0 + (np.int(n) - 0.5) * np.float(dx), decimals=d),
    )
    return np.amin(coords), np.amax(coords)

# --------Unet Tier1----------------------------------------------------------------

# process netcdof file => processing
def process_one_file(
    nc_file_in: str, grid_des_tar: dict, grid_des_coarse: dict, grid_des_base: dict
):
    """
    Preprocess one netCDF-datafile.
    :param nc_file_in: input netCDF-file to be preprocessed.
    :param grid_des_tar: dictionary for grid description of target data
    :param grid_des_coarse: dictionary for grid description of coarse data
    :param grid_des_base: dictionary for grid description of auxiliary data
    """

    # sanity check
    if not os.path.isfile(nc_file_in):
        raise FileNotFoundError(f"%Could not find netCDF-file '{nc_file_in}'.")
    # hard-coded constants [IFS-specific parameters (from Chapter 12 in http://dx.doi.org/10.21957/efyk72kl)]
    cpd, g = 1004.709, 9.80665
    # get path to grid description files
    kf = "file"
    fgrid_des_base, fgrid_des_coarse, fgrid_des_tar = (
        grid_des_base[kf],
        grid_des_coarse[kf],
        grid_des_tar[kf],
    )
    # get parameters
    lon0_b, lon1_b = get_slice_coords(
        grid_des_base["xfirst"], grid_des_base["xinc"], grid_des_base["xsize"]
    )
    lat0_b, lat1_b = get_slice_coords(
        grid_des_base["yfirst"], grid_des_base["yinc"], grid_des_base["ysize"]
    )
    lon0_tar, lon1_tar = get_slice_coords(
        grid_des_tar["xfirst"], grid_des_tar["xinc"], grid_des_tar["xsize"]
    )
    lat0_tar, lat1_tar = get_slice_coords(
        grid_des_tar["yfirst"], grid_des_tar["yinc"], grid_des_tar["ysize"]
    )
    # initialize tools
    cdo, ncrename, ncap2, ncks, ncea = CDO(), NCRENAME(), NCAP2(), NCKS(), NCEA()

    fname_base = nc_file_in.rstrip(".nc")

    # start processing chain
    # slice data to region of interest and relevant lead times
    nc_file_sd = fname_base + "_subdomain.nc"
    ncea.run(
        [nc_file_in, nc_file_sd],
        OrderedDict(
            [
                ("-O", ""),
                (
                    "-d",
                    [
                        "time,0,11",
                        "latitude,{0},{1}".format(lat0_b, lat1_b),
                        "longitude,{0},{1}".format(lon0_b, lon1_b),
                    ],
                ),
            ]
        ),
    )

    ncrename.run(
        [nc_file_sd],
        OrderedDict(
            [
                ("-d", ["latitude,lat", "longitude,lon"]),
                ("-v", ["latitude,lat", "longitude,lon"]),
            ]
        ),
    )
    ncap2.run(
        [nc_file_sd, nc_file_sd],
        OrderedDict([("-O", ""), ("-s", '"lat=double(lat); lon=double(lon)"')]),
    )

    # calculate dry static energy fir first-order conservative remapping
    nc_file_dse = fname_base + "_dse.nc"
    ncap2.run(
        [nc_file_sd, nc_file_dse],
        OrderedDict(
            [("-O", ""), ("-s", '"s={0}*t2m+z+{1}*2"'.format(cpd, g)), ("-v", "")]
        ),
    )
    # add surface geopotential to file
    ncks.run([nc_file_sd, nc_file_dse], OrderedDict([("-A", ""), ("-v", "z")]))

    # remap the data (first-order conservative approach)
    nc_file_crs = fname_base + "_coarse.nc"
    cdo.run(
        [nc_file_dse, nc_file_crs],
        OrderedDict([("remapcon", fgrid_des_coarse), ("-setgrid", fgrid_des_base)]),
    )

    # remap with extrapolation on the target high-resolved grid with bilinear remapping
    nc_file_remapped = fname_base + "_remapped.nc"
    cdo.run(
        [nc_file_crs, nc_file_remapped],
        OrderedDict([("remapbil", fgrid_des_tar), ("-setgrid", fgrid_des_coarse)]),
    )
    # retransform dry static energy to t2m
    ncap2.run(
        [nc_file_remapped, nc_file_remapped],
        OrderedDict(
            [("-O", ""), ("-s", '"t2m_in=(s-z-{0}*2)/{1}"'.format(g, cpd)), ("-o", "")]
        ),
    )
    # finally rename data to distinguish between input and target data
    # (the later must be copied over from previous files)
    ncrename.run([nc_file_remapped], OrderedDict([("-v", "z,z_in")]))
    ncks.run(
        [nc_file_remapped, nc_file_remapped],
        OrderedDict([("-O", ""), ("-x", ""), ("-v", "s")]),
    )
    # NCEA-bug with NCO/4.9.5: Add slide offset to lon1_tar to avoid corrupted data in appended file
    # (does not affect slicing at all)
    lon1_tar = lon1_tar + np.float(grid_des_tar["xinc"]) / 10.0
    ncea.run(
        [nc_file_sd, nc_file_remapped],
        OrderedDict(
            [
                ("-A", ""),
                (
                    "-d",
                    [
                        "lat,{0},{1}".format(lat0_tar, lat1_tar),
                        "lon,{0},{1}".format(lon0_tar, lon1_tar),
                    ],
                ),
                ("-v", "t2m,z"),
            ]
        ),
    )
    ncrename.run([nc_file_remapped], OrderedDict([("-v", ["t2m,t2m_tar", "z,z_tar"])]))

    if os.path.isfile(nc_file_remapped):
        print(
            f"%Processed data successfully from '{nc_file_in}' to '{nc_file_remapped}'. Cleaning-up..."
        )
        for f in [nc_file_sd, nc_file_dse, nc_file_crs]:
            os.remove(f)
    else:
        raise RuntimeError(
            f"%Something went wrong when processing '{nc_file_in}'. Check intermediate files."
        )

    return True

# --------------------------------ERA5-IFS-------------------------------------------------------------------

# gather unique pressure levels
def retrieve_plvls(mlvars_dict):
    """
    Returns list of unique pressure levels from nested variable dictionary of form
    :param mlvars_dict: nested variable dictionary, e.g. {<var1>: ["p85000", "p92500"], <var2>: ["p85000"]}
    :return: list of uniues pressure levels, e.g [85000, 925000] in this example
    """
    lvls = set(list(flatten(mlvars_dict.values())))
    plvls = [int(float(lvl.lstrip("p"))) for lvl in lvls if lvl.startswith("p")]
    # Currently only pressure-level interpolation is supported. Thus, we stop here if other level identifier is used
    if len(lvls) != len(plvls):
        raise ValueError("Could not retrieve all parsed level imformation. Check the folllowing: {0}"
                        .format(", ".join(lvls)))

    return plvls


# sort variables by type => handle in variable class
def organize_predictors(predictors: dict) -> Tuple[List, dict, List, dict]:
    """
    Checks predictors for variables to process and returns condensed information for further processing
    :param predictors: dictionary for predictors
    :return: list of surface and forecast variables and dictionary of multi-level variables to interpolate
    """
    known_vartypes = ["sf", "ml", "fc_sf", "fc_pl"]

    pred_vartypes = list(predictors.keys())
    lpred_vartypes = [pred_vartype in known_vartypes for pred_vartype in pred_vartypes]
    if not all(lpred_vartypes):
        unknown_vartypes = [
            pred_vartypes[i] for i, flag in enumerate(lpred_vartypes) if not flag
        ]
        raise ValueError(
            f"%The following variables types in the predictor-dictionary are unknown: {', '.join(unknown_vartypes)}"
        )

    sfvars, mlvars, fc_sfvars, fc_plvars = (
        predictors.get("sf", None),
        predictors.get("ml", None),
        predictors.get("fc_sf", None),
        predictors.get("fc_pl", None),
    )

    # some checks (level information redundant for surface-variables)
    if sfvars:
        if any([i is not None for i in sfvars.values()]):
            print(
                "%Some values of sf-variables are not None, but do not have any effect."
            )
        sfvars = list(sfvars)

    if fc_sfvars:
        if any([i is not None for i in fc_sfvars.values()]):
            print(
                "%Some values of fc_sf-variables are not None, but do not have any effect."
            )
        fc_sfvars = list(fc_sfvars)

    if mlvars:
        mlvars["plvls"] = retrieve_plvls(mlvars)

    if fc_plvars:
        fc_plvars["plvls"] = retrieve_plvls(fc_plvars)

    return sfvars, mlvars, fc_sfvars, fc_plvars

# (supress exceptions until max exceptions is exceeded => maybe decorator ?)
def run_preproc_func(
    preproc_func: callable,
    args: List,
    kwargs: dict,
    logger: logging.Logger,
    nwarns: int,
    max_warns: int,
) -> Tuple[int, str]:
    """
    Run a function where arguments are parsed from list. Counts failures as warnings unless max_warns is exceeded
    or the error is not a Runtime-Error
    :param preproc_func: the callable preprocessing-function
    :param args: list of arguments to be parsed to preproc_func
    :param kwargs: dictionary of keyword arguments to be parsed to preproc_func
    :param logger: logger instance
    :param nwarns: current number of issued warnings
    :param max_warns: maximum allowed number of warnings
    :return: updated nwarns and outfile
    """
    assert callable(preproc_func), "func is not a callable, but of type '{0}'".format(
        type(preproc_func)
    )

    try:
        outfile = preproc_func(*args, **kwargs)
    except (RuntimeError, FileNotFoundError) as err:
        mess = "Pre-Processing data from '{0}' failed! ".format(args[0])
        nwarns += 1
        if nwarns > max_warns:
            logger.fatal(mess + "Maximum number of warnings exceeded.")
            raise err
        else:
            logger.error(mess), logger.error(str(err))
            outfile = None
    except BaseException as err:
        logger.fatal(
            "Something unexpected happened when handling data from '{0}'. See error-message".format(
                args[0]
            )
        )
        raise err

    return nwarns, outfile

# append to name to list or delete list/associated files => burn it with fire
def manage_filemerge(
    filelist: List, file2merge: str, tmp_dir: str, search_patt: str = "*.nc"
):
    """
    Add file2merge to list of files or clean-up temp-dirctory if file2merge is None
    :param filelist: list of files to be updated
    :param file2merge: file to merge
    :param tmp_dir: directory for temporary data
    :param search_patt: search pattern for files to remove
    :return: updated filelist
    """
    if file2merge:
        filelist.append(file2merge)
    else:
        remove_list = glob.iglob(os.path.join(tmp_dir, search_patt))
        remove_files(remove_list, lbreak=True)
        filelist = []
    return filelist

# sort variables by dynamic/static => handle in variable class
# filter(lambda var: var.type == "surface", domain.variables)
def split_dyn_static(sfvars: List):
    """
    Split list of surface variables into lists of static and dynamical variables (see const_vars-variable of class).
    :param sfvars: input list of surface variables
    :return: two lists where the first holds the static and the second holds the dynamical variables
    """
    sfvars_stat = [sfvar for sfvar in sfvars if sfvar in ["z", "lsm"]]
    sfvars_dyn = [sfvar for sfvar in sfvars if sfvar not in sfvars_stat]

    return sfvars_stat, sfvars_dyn

# get ecmwf forcast file for date => time domain ? (!era5/ifs specific) => maybe implemnt interface/protocoll
def get_fc_file(
    dirin_base: str,
    date: dt.datetime,
    offset: int = 6,
    model: str = "era5",
    suffix="",
    prefix="",
) -> Tuple[str, int]:
    """
    Construct path to forecast file corresponding to specific date from ECMWF forecasts (e.g. IFS or ERA5).
    :param dirin_base: top-level directory where ECMWF forecasts are placed (in <year>/<year>-<month>/-subdirs)
    :param date: The date for which forecast data is requested
    :param model: The ECMWF model for which forecast file is requested (either 'ERA5' or 'IFS')
    :param offset: Offset in hours for forecasts to use (e.g. 6 means that lead times smaller 6 hours are not used)
    :param suffix: Suffix to forecast filename (for IFS-forecasts only, e.g. 'sfc' or 'pl')
    :param prefix: Prefix to forecast filename (for ERA5-forecasts only, e.g. 'sf_fc' or 'pl_fc')
    :return: path to corresponding forecast file
    """
    # sanity checks and setting of model initialization time
    assert offset < 12, "Offset must be smaller than 12, but is {0:d}".format(offset)

    model = model.lower()
    if model == "era5":
        init_model = [6, 18]
    elif model == "ifs":
        init_model = [0, 12]
    else:
        raise ValueError(
            "Model {0} is not supported. Only IFS and ERA5 are valid models.".format(
                model
            )
        )
    # get daytime hour
    hour = int(date.strftime("%H"))

    # construct initialization time of model run and corresponding forecast hour
    if hour < offset + init_model[0]:
        fh = 24 - init_model[1] + hour
        run_init = date.replace(hour=init_model[1]) - dt.timedelta(days=1)
    elif offset + init_model[0] <= hour < offset + init_model[1]:
        fh = hour - init_model[0]
        run_init = date.replace(hour=init_model[0])
    elif hour >= init_model[1] + offset and init_model[1] + offset < 24:
        fh = hour - init_model[1]
        run_init = date.replace(hour=init_model[1])
    else:
        raise ValueError(
            "Combination of init hours ({0:d}, {1:d}) and offset {2} not implemented.".format(
                init_model[0], init_model[1], offset
            )
        )
    # construct resulting filenames
    nc_file = ""
    if model == "era5":
        nc_file = os.path.join(
            dirin_base,
            run_init.strftime("%Y"),
            run_init.strftime("%m"),
            "fc_{0}".format(run_init.strftime("%H")),
            "{0}_{1:d}00_{2:d}_{3}.grb".format(
                run_init.strftime("%Y%m%d"), int(run_init.strftime("%H")), fh, prefix
            ),
        )
    elif model == "ifs":
        nc_file = os.path.join(
            dirin_base,
            run_init.strftime("%Y"),
            run_init.strftime("%Y-%m"),
            "{0}_{1}_{2}.nc".format(
                suffix, run_init.strftime("%Y%m%d"), run_init.strftime("%H")
            ),
        )

    if not os.path.isfile(nc_file):
        raise FileNotFoundError(
            "Could not find requested forecast file '{0}'".format(nc_file)
        )

    return nc_file, fh


# -------------------------ERA5-CREA6---------------------------------

# redundant to "organize_predictors"
def organize_predictands(predictands: dict) -> Tuple[List, List]:
    """
    Organizes predictands from COSMO-REA6 dataset. Currently, only 2D variables and invariant data are supported.
    !!! To-Do !!!
    3D variables incl. interpolation (on pressure-levels) has to be integrated
    !!! To-Do !!
    :param predictands: dictionary for predictands with the form {"2D", {"t_2m"}}
    """
    known_vartypes = ["2D", "const"]

    pred_vartypes = list(predictands.keys())
    lpred_vartypes = [pred_vartype in known_vartypes for pred_vartype in pred_vartypes]
    if not all(lpred_vartypes):
        unknown_vartypes = [
            pred_vartypes[i] for i, flag in enumerate(lpred_vartypes) if not flag
        ]
        raise ValueError(
            "The following variables types in the predictands-dictionary are not supported: {0}".format(
                ", ".join(unknown_vartypes)
            )
        )

    vars_2d, vars_const = predictands.get("2D", None), predictands.get("const", None)
    vars_2d = [var_2d.upper() for var_2d in vars_2d]
    vars_const = [var_const.upper() for var_const in vars_const]

    return vars_2d, vars_const

# is reqiered domain prsent for variable in dataset => handle by pydantic
def check_crea6_files(
    indir: str, const_file: str, yr_month: str, vars_2d: List, const_vars: List
):
    """
    Checks if required predictands are available from the COSMO-REA6 dataset.
    Data is expected to live in monthly netCDF-files within subdirectories named as the variable of interest.
    For invariant data, the information is mandatory in the invariant file of the COSMO-REA6 dataset.
    :param indir: Directory under which COSMO-REA6 data is stored
    :param const_file: path to invariant datafile
    :param yr_month: Date-string for which data is required; format YYYY-MM
    :param vars_2d: list of 2D-variables serving as predictands
    :param const_vars: list of invariant variables serving as predictands (must be part of const_file)
    """
    if vars_2d:
        for var_2d in vars_2d:
            var_2d = var_2d.capitalize()
            dfile_2d = os.path.join(indir, "2D", var_2d, f"{var_2d}.2D.{yr_month}")
            if not os.path.isfile(dfile_2d):
                FileNotFoundError(
                    f"Could not find required file '{dfile_2d}' for predictand variable '{dfile_2d}'"
                )

    if const_vars:
        fconsts = xr.open_dataset(const_file)
        varlist = list(fconsts.keys())
        stat = [cvar in varlist for cvar in const_vars]
        if not all(stat):
            miss_inds = np.where(not stat)
            raise ValueError(
                "The following variables cannot be found in the constant file '{0}': {1}".format(
                    const_file, ",".join(list(np.array(const_vars)[miss_inds]))
                )
            )

    return True
