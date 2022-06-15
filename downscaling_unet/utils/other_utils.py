__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-01-20"
__update__ = "2022-06-15"

import os
import inspect
import numpy as np
import pandas as pd
import datetime as dt
from dateutil.parser import parse as date_parser
from typing import Any, List
try:
    from collections import Iterable
except ImportError:
    from typing import Iterable

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
"""
# doc-string


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
    Remove single key from dictionary if it is present. Returns a new dict.
    :param dict_in: input dictionary
    :param key: key to be removed
    """
    return {k: v for k, v in dict_in.items() if k != key}


def to_list(obj: Any) -> List:
    """
    Method from MLAIR!
    Transform given object to list if obj is not already a list. Sets are also transformed to a list.
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
    Subsets a list of files based on a time-pattern that must be part of the filename.
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
    method = ensure_datetime.__name__

    if isinstance(date, dt.datetime):
        date_dt = date
    else:
        try:
            date_dt = pd.to_datetime(date).to_pydatetime()
        except Exception as err:
            print("%{0}: Could not handle input date (as string: {1}, type: {2}).".format(method, str(date), type(date)))
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
    Yield items from any nested iterable.
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
