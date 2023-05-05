# SPDX-FileCopyrightText: 2023 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

# doc-string
"""
Some auxiliary functions for the project:
    * provide_default
    * remove_key_from_dict
    * to_list
    * get_func_kwargs
    * subset_files_on_date
    * extract_date
    * ensure_datetime
    * last_day_of_month
    * flatten
    * remove_files
    * check_str_in_list
    * find_closest_divisor
    * free_mem
    * print_gpu_usage
    * print_cpu_usage
    * get_memory_usage
    * get_max_memory_usage
    * copy_filelist
"""
# doc-string

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-01-20"
__update__ = "2023-03-17"

import os
import gc
import inspect
import psutil
import resource
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime as dt
from dateutil.parser import parse as date_parser
import shutil
from typing import Any, List, Union
try:
    from collections import Iterable
except ImportError:
    from typing import Iterable

str_or_List = Union[List, str]


def provide_default(dict_in, keyname, default=None, required=False):
    """
    Returns values of key from input dictionary or alternatively its default
    :param dict_in: input dictionary
    :param keyname: name of key which should be added to dict_in if it is not already existing
    :param default: default value of key (returned if keyname is not present in dict_in)
    :param required: Forces existence of keyname in dict_in (otherwise, an error is returned)
    :return: value of requested key or its default retrieved from dict_in
    """

    if not required and default is None:
        raise ValueError("Provide default when existence of key in dictionary is not required.")

    if keyname not in dict_in.keys():
        if required:
            print(dict_in)
            raise ValueError("Could not find '{0}' in input dictionary.".format(keyname))
        return default
    else:
        return dict_in[keyname]


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


def remove_files(files: List, lbreak: True):
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


def free_mem(var_list: List):
    """
    Delete all variables in var_list and release memory
    :param var_list: list of variables to be deleted
    """
    var_list = to_list(var_list)
    for var in var_list:
        del var

    gc.collect()

# The following auxiliary methods have been adapted from MAELSTROM AP3,
# see https://git.ecmwf.int/projects/MLFET/repos/maelstrom-radiation/browse/climetlab_maelstrom_radiation/benchmarks/
#                           utils.py?at=jube


def print_gpu_usage(message="", show_line=False):
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


def copy_filelist(file_list: List, dest_dir: str, file_list_dest: List: None ,labort: bool = True):
    """
    Copy a list of files to another directory
    :param file_list: list of files to copy
    :param dest_dir: target directory to which files will be copied
    :param labort: flag to trigger raising of an error (if False, only Warning-messages will be printed)
    """
    file_list = to_list(file_list)
    if not os.path.isdir(dest) and labort:
        raise NotADirectoryError(f"Cannot copy to non-existing directory '{dest}'.")
    elif not os.path.isdir(dest) and not labort:
        print(f"WARNING: Target directory for copying '{dest}' does not exist. Skip copy process...")

    if file_list_dest is None:
        dest = dest_dir
    else:
        assert len(file_list) == len(file_list_dest), f"Length of filelist to copy ({len(file_list)})" + \
                                                      f" and of filelist at destination ({len(file_list_dest)}) differ."
        dest = [os.path.join(dest_dir, f_dest) for f_dest in file_list_dest]

    for f in file_list:
        if os.path.isfile(f):
            shutil.copy(f, dest)
        else:
            if labort:
                raise FileNotFoundError(f"Could not find file '{f}'. Error will be raised.")
            else:
                print(f"WARNING: Could not find file '{f}'.")