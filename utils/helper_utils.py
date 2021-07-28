# ********** Info **********
# @Creation: 2021-07-28
# @Update: 2021-07-28
# @Author: Michael Langguth
# @Site: Juelich supercomputing Centre (JSC) @ FZJ
# @File: helper.py
# ********** Info **********

"""
A collection of auxiliary functions.

The following functions are provided:
    * ensure_datetime
    * extract_date
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

