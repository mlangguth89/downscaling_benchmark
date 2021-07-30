# ********** Info **********
# @Creation: 2021-07-28
# @Update: 2021-07-30
# @Author: Michael Langguth
# @Site: Juelich supercomputing Centre (JSC) @ FZJ
# @File: helper.py
# ********** Info **********

"""
A collection of auxiliary functions.

The following functions are provided:
    * ensure_datetime
    * extract_date
    * subset_files_on_date
"""

import os
import numpy as np
import pandas as pd
import datetime as dt
from dateutil.parser import parse as date_parser


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