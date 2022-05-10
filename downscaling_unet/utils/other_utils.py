__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-01-20"
__update__ = "2022-05-02"

import os
import inspect
from typing import Any, List
try:
    from collections import Iterable
except ImportError:
    from typing import Iterable
import datetime as dt

# doc-string
"""
Some auxiliary functions for the project:
    * provide_default
    * remove_key_from_dict
    * to_list
    * get_func_kwargs
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
