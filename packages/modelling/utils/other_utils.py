# SPDX-FileCopyrightText: 2025 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC); Gesosphere Austria (GSA)
#
# SPDX-License-Identifier: MIT

# doc-string
"""
Some auxiliary functions for the project:
    * config_logger
    * remove_key_from_dict
    * to_list
    * get_func_kwargs
    * subset_files_on_date
    * extract_date
    * ensure_datetime
    * doy_to_mo
    * last_day_of_month
    * flatten
    * remove_files
    * check_str_in_list
    * check_and_get_vars_from_ds
    * shape_from_str
    * find_closest_divisor
#    * free_mem
    * print_gpu_usage
    * print_cpu_usage
    * get_memory_usage
    * get_max_memory_usage
    * copy_filelist
    * merge_dicts
    * finditem
    * remove_items
    * convert_to_xarray
    * get_training_time_dict
    * get_batch_size_mb
"""
# doc-string

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-01-20"
__update__ = "2024-03-25"

import os, sys
import gc
import inspect
import psutil
import resource
import logging
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
from dateutil.parser import parse as date_parser
import shutil
from typing import Any, List, Union, Dict, Tuple
try:
    from collections import Iterable
except ImportError:
    from typing import Iterable
from skimage.util.shape import view_as_blocks

str_or_List = Union[List, str]
logger_module_name = f"other_utils.{__name__}"


def config_logger(logger, logfile: str, log_level_file=logging.DEBUG, log_level_console=logging.INFO, remove_existing_file: bool = True):

    if remove_existing_file and os.path.isfile(logfile):
        os.remove(logfile)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: %(message)s')

    fh = logging.FileHandler(logfile)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(log_level_console)
    fh.setLevel(log_level_file)

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh), logger.addHandler(ch)

    return logger

def remove_key_from_dict(dict_in: dict, key: str) -> dict:
    """
    Remove single key from dictionary if it is present. Returns a new dict
    :param dict_in: input dictionary
    :param key: key to be removed
    """
    return {k: v for k, v in dict_in.items() if k != key}


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


def get_func_kwargs(func, kwargs):
    """
    Returns dictonary of keyword arguments that can be used for method
    :param func: callable method
    :param kwargs: dictionary of keyword arguments from which to extract keyword arguments of interest
    :return: method_kwargs
    """
    func_args = list(inspect.signature(func).parameters)
    func_kwargs = {k: kwargs.pop(k) for k in dict(kwargs) if k in func_args}

    return func_kwargs


def subset_files_on_date(all_files_list: list, val: int, filter_basedir: bool = False, date_alias: str = "H"):
    """
    Subsets a list of files based on a time-pattern that must be part of the filename
    :param all_files_list: list of all files
    :param val: time value (default meaning: hour of the day, see date_alias)
    :param filter_basedir: flag for removing base-directory when subsetting, e.g. when dates are present in basedir
    :param date_alias: also known as offset alias in pandas
    (see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases)
    """
    method = subset_files_on_date.__name__

    if filter_basedir:
        all_files_dates = [(extract_date(os.path.dirname(dfile))).strftime(date_alias) for dfile in all_files_list]
    else:
        all_files_dates = [(extract_date(dfile)).strftime(date_alias) for dfile in all_files_list]
    inds = [idx for idx, s in enumerate(all_files_dates) if "{0:02d}".format(int(val)) in s]

    if not inds:
        raise ValueError("%{0}: Could not find any file carrying the value of {1:02d} with date alias {2}"
                         .format(method, val, date_alias))
    else:
        return list(np.asarray(all_files_list)[inds])


def extract_date(date_str):
    """
    Checks if a datetime-object can be extracted from a given string.
    Based on dateutil.parser.parse.
    :param date_str: Any string containing some date
    :return: A corresponding datetime object
    """
    method = extract_date.__name__

    assert isinstance(date_str, str), "Input must be a string."
    try:
        date_extracted = date_parser(date_str, fuzzy=True)
    except Exception as err:
        print("%{0}: Could not extract date from '{1}'. Investigate raised error".format(method, date_str))
        raise err
    return date_extracted


def ensure_datetime(date):
    """
    Tries to convert date which can be everything that can be processed by pandas' to_datetime-method
    into a datetime.datetime-object.
    :param date: Any date that can be handled by to_datetime
    :param: Same as date, but as datetime.datetime-onject
    """
    if isinstance(date, dt.datetime):
        date_dt = date
    else:
        try:
            date_dt = pd.to_datetime(date).to_pydatetime()
        except Exception as err:
            print("Could not handle input date (as string: {0}, type: {1}).".format(str(date), type(date)))
            raise err

    return date_dt


def doy_to_mo(day_of_year: int, year: int):
    """
    Converts day of year to year-month datetime object (e.g. in_day = 2, year = 2017 yields January 2017).
    From AtmoRep-project: https://isggit.cs.uni-magdeburg.de/atmorep/atmorep/
    :param day_of_year: day of year
    :param year: corresponding year (to reflect leap years)
    :return year-month datetime object
    """
    date_month = pd.to_datetime(year * 1000 + day_of_year, format='%Y%j')
    return date_month


def last_day_of_month(any_day):
    """
    Returns the last day of a month
    :param any_day : datetime object with any day of the month
    :return: datetime object of lat day of month
    """
    next_month = any_day.replace(day=28) + dt.timedelta(days=4)  # this will never fail
    return next_month - dt.timedelta(days=next_month.day)


def flatten(nested_iterable):
    """
    Yield items from any nested iterable
    :return Any nested iterable.
    """
    for x in nested_iterable:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x


def remove_files(files: List, lbreak: bool = True):
    """
    Remove files from a list
    :param files: list of file names
    :param lbreak: flag of error is risen if non-existing files are encountered
    :return: -
    """
    for file in files:
        if os.path.isfile(file):
            os.remove(file)
        else:
            mess = "File '{0}' does not exist and thus cannot be removed.".format(file)
            if lbreak:
                raise ValueError(mess)
            else:
                print(mess)


def check_str_in_list(list_in: List, str2check: str_or_List, labort: bool = True, return_ind: bool = False):
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
        assert np.all([isinstance(str1, str) for str1 in str2check]), "Not all elements of str2check are strings"
    else:
        raise ValueError("str2check-argument must be either a string or a list of strings")

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
    
def check_and_get_vars_from_ds(ds, var_list: List[str], suffix: str = "*"):
    """
    Checks if all variables in var_list are part of the dataset. If var_list is None, all variables with a given
    suffix are returned.
    :param ds: The xarray.Dataset 
    :param var_list: list of predictor variables or None
    :param suffix: optional suffix of variables to selected. Only effective if var_list is None
    :return selected_vars: list of selected variables
    """
    data_vars = list(ds.variables)

    if var_list is None:
        selected_vars = [var for var in data_vars if var.endswith(suffix)]
    else:
        stat_list = [var in data_vars for var in var_list]
        if all(stat_list):
            selected_vars = var_list
        else:
            miss_inds = [i for i, x in enumerate(stat_list) if not x]
            miss_vars = [var_list[i] for i in miss_inds]
            raise ValueError(f"Could not find the following variables in the dataset: {*miss_vars,}")

    return selected_vars


def shape_from_str(fname):
    """
    Retrieves shapes from AtmoRep output-filenames.
    From AtmoRep-project: https://isggit.cs.uni-magdeburg.de/atmorep/atmorep/
    :param fname: filename of AtmoRep output-file
    :return shapes inferred from AtmoRep output-file
    """
    shapes_str = fname.replace("_phys.dat", ".dat").split(".")[-2].split("_s_")[-1].split("_") #remove .ext and split
    shapes = [int(i) for i in shapes_str]
    return shapes

    
    
def find_closest_divisor(n1, div):
    """
    Function to find closest divisor for a given number with respect to a target value
    :param n1: The number for which a divisor should be found
    :param div: The desired divisor value
    :return div_new: In case that div is a divisor n1, div remains unchanged. In any other case,
                     the closest integer to div is returned.
    """
    def get_divisors(n):
        res = []
        i = 1
        while i <= n:
            if n % i == 0:
                res.append(i),
            i += 1
        return res

    all_divs = get_divisors(n1)

    if div in all_divs:
        return div
    else:
        i = np.argmin(np.abs(np.array(all_divs) - div))
        return all_divs[i]


#def free_mem(var_list: List):
# *** This was found to be in effective, cf. michael_issue085-memory_footprint ***
#
#    """
#    Delete all variables in var_list and release memory
#    :param var_list: list of variables to be deleted
#    """
#    var_list = to_list(var_list)
#    for var in var_list:
#        del var
#
#    gc.collect()

# The following auxiliary methods have been adapted from MAELSTROM AP3,
# see https://git.ecmwf.int/projects/MLFET/repos/maelstrom-radiation/browse/climetlab_maelstrom_radiation/benchmarks/
#                           utils.py?at=jube


def print_gpu_usage(message="", show_line=False):
    
    # This method is only available if tensorflow is installed
    import tensorflow as tf

    try:
        usage = tf.config.experimental.get_memory_info("GPU:0")
        output = message + ' - '.join([f"{k}: {v / 1024**3:.2f} GB" for k, v in usage.items()])
    except ValueError as _:
        output = message + ' None'

    if show_line:
        frameinfo = inspect.getouterframes(inspect.currentframe())[1]
        output += " (%s:%s)" % (frameinfo.filename, frameinfo.lineno)

    print(output)


def print_cpu_usage(message: str = "", show_line: bool = False):
    """
    Prints the current and maximum memory useage of this process
    :param message: Prepend with this message
    :param show_line: Add the file and line number making this call at the end of message
    """

    output = "current: %.2f GB - peak: %.2f GB" % (
        get_memory_usage() / 1024 ** 3,
        get_max_memory_usage() / 1024 ** 3,
    )
    output = message + output
    if show_line:
        frameinfo = inspect.getouterframes(inspect.currentframe())[1]
        output += " (%s:%s)" % (frameinfo.filename, frameinfo.lineno)

    print(output)


def get_memory_usage():
    p = psutil.Process(os.getpid())
    mem = p.memory_info().rss
    for child in p.children(recursive=True):
        mem += child.memory_info().rss
    return mem


def get_max_memory_usage():
    """In bytes"""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1000


def copy_filelist(file_list: List, dest_dir: str, file_list_dest: List = None ,labort: bool = True):
    """
    Copy a list of files to another directory
    :param file_list: list of files to copy
    :param dest_dir: target directory to which files will be copied
    :param labort: flag to trigger raising of an error (if False, only Warning-messages will be printed)
    """
    file_list = to_list(file_list)
    if not os.path.isdir(dest_dir) and labort:
        raise NotADirectoryError(f"Cannot copy to non-existing directory '{dest}'.")
    elif not os.path.isdir(dest_dir) and not labort:
        print(f"WARNING: Target directory for copying '{dest}' does not exist. Skip copy process...")
        return

    if file_list_dest is None:
        dest = dest_dir
    else:
        assert len(file_list) == len(file_list_dest), f"Length of filelist to copy ({len(file_list)})" + \
                                                      f" and of filelist at destination ({len(file_list_dest)}) differ."
        dest = [os.path.join(dest_dir, f_dest) for f_dest in file_list_dest]

    for i, f in enumerate(file_list):
        if os.path.isfile(f):
            shutil.copy(f, dest[i])
        else:
            if labort:
                raise FileNotFoundError(f"Could not find file '{f}'. Error will be raised.")
            else:
                print(f"WARNING: Could not find file '{f}'.")

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
                if isinstance(value, (list, tuple)) and isinstance(default_value, (list, tuple)):
                    value = tuple(value) if isinstance(default_value, tuple) else list(value)

                assert isinstance(value, expected_type), \
                    f"Type mismatch for key '{key}': Expected {expected_type}, got {type(value)}"

            merged_dict[key] = value

    return merged_dict

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


def remove_items(obj: Union[List, Dict, Tuple], items: Any):
    """
    Remove item(s) from either list, tuple or dictionary.

    :param obj: object to remove items from (either dictionary or list)
    :param items: elements to remove from obj. Can either be a list or single entry / key

    :return: object without items
    """

    def remove_from_list(list_obj, item_list):
        """Remove implementation for lists."""
        if len(item_list) > 1:
            return [e for e in list_obj if e not in item_list]
        elif len(item_list) == 0:
            return list_obj
        else:
            list_obj = list_obj.copy()
            try:
                list_obj.remove(item_list[0])
            except ValueError:
                pass
            return list_obj

    def remove_from_dict(dict_obj, key_list):
        """Remove implementation for dictionaries."""
        return {k: v for k, v in dict_obj.items() if k not in key_list}

    items = to_list(items)
    if isinstance(obj, list):
        return remove_from_list(obj, items)
    elif isinstance(obj, dict):
        return remove_from_dict(obj, items)
    elif isinstance(obj, tuple):
        return tuple(remove_from_list(to_list(obj), items))
    else:
        raise TypeError(f"{inspect.stack()[0][3]} does not support type {type(obj)}.")
    
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
        if isinstance(mout_np, list): mout_np = mout_np[0]
        mout_xr = xr.DataArray(mout_np[..., 0].squeeze(), coords=coords, dims=dims, name=varname)
    else:
        # no slicing required
        mout_xr = xr.DataArray(mout_np.squeeze(), coords=coords, dims=dims, name=varname)

    # get variable-specific normalizer and perform denormalization
    norm_var = norm.get_normalizer_for_var(varname)

    mout_xr = norm_var.denormalize(mout_xr, varname=varname)

    return mout_xr

def get_training_time_dict(epoch_times: list, steps):
    """
    Computes training times from a list of epoch times
    :param epoch_times: list of epoch times obtained from TimeHistory-callback (see model_utils.py)
    :param steps: number of steps
    """
    tot_time = np.sum(epoch_times)

    training_times = {"Total training time": np.sum(epoch_times), "Avg. training time per epoch": np.mean(epoch_times),
                      "Min. training time per epoch": np.amin(epoch_times),
                      "Max. training time per epoch": np.amax(epoch_times[1:]),
                      "First epoch training time": epoch_times[0], "Avg. training time per iteration": tot_time/steps}

    return training_times

def get_batch_size_mb(shape_in, batch_size):
    """
    Computes the memory footprint of a batch of data
    :param shape_in: shape of a single data sample
    :param batch_size: batch size
    :return: memory footprint of a batch of data
    """
    return np.prod(shape_in) * batch_size * 4 / 1.e+06

def sample_permut_xyt(da_orig: xr.DataArray, patch_size:tuple = (8, 8)):
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

    # revert view_as_blocks-operation
    da_patched = da_patched.values.transpose([0, 3, 1, 4, 2, 5]).reshape(sh_aux)
    
    # write data back on da_permute
    da_permute[...] = np.moveaxis(da_patched, 2, 0)

    # transpose to original dimension ordering if required
    da_permute = da_permute.transpose(*dims_orig)

    return da_permute
