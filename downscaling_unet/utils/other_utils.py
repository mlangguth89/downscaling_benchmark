__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-01-20"
__update__ = "2022-01-22"

import inspect
from typing import Any, List

# doc-string
"""
Some auxiliary functions for the project.
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


def griddes_lines_to_dict(lines):
    """
    Converts the lines that were read from a CDO grid description file to a dictionary.
    The lines must follow the convention '<key> = <value>' to be recognized. Other lines will be ignored
    :param lines: lines from grid description
    :return: dictionary carrying keys with corresponding values as string from lines of grid description file.
    """
    dict_out = {}

    lines = to_list(lines)
    for line in lines:
        splitted = line.replace("\n", "").split("=")
        if len(splitted) == 2:
            dict_out[splitted[0].strip()] = splitted[1].strip()

    return dict_out


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
