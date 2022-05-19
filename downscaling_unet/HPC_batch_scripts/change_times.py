import xarray as xr
import pandas as pd
import datetime as dt

def last_day_of_month(any_day):
    """
    Returns the last day of a month
    :param any_day : datetime object with any day of the month
    :return: datetime object of lat day of month
    """
    next_month = any_day.replace(day=28) + dt.timedelta(days=4)  # this will never fail
    return next_month - dt.timedelta(days=next_month.day)


date_start=dt.datetime.strptime("2018-02", "%Y-%m")
dates_start=pd.date_range(date_start, dt.datetime.strptime("2018-02-01", "%Y-%m-%d"), freq="MS")

for date_start in dates_start:
    date_end=last_day_of_month(date_start).replace(hour=23)
    times_new=pd.date_range(date_start+dt.timedelta(hours=23), date_end, freq="H")
    dfile="/p/scratch/deepacf/maelstrom/maelstrom_data/ap5_michael/preprocessed_era5_ifs/netcdf_data/{0}/{1}/preproc_{1}.nc".format(date_start.strftime("%Y"), date_start.strftime("%Y-%m"))
    with xr.open_dataset(dfile) as d:
        d_copy = d.copy()
    d_copy["time"]=times_new
    print("Write new times to '{0}'".format(dfile))
    d_copy.to_netcdf(dfile, engine="scipy")
