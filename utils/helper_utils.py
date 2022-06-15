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


