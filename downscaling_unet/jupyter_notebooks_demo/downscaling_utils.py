import os, sys
from timeit import default_timer as timer
import datetime as dt
# for processing data
import numpy as np
import xarray as xr
import pandas as pd
# for plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
# make class for handling data available
sys.path.append("../")
from input_data_class import InputDataClass

# class for getting data


class DownscalingData(InputDataClass):

    def __init__(self, datadir: str, app: str = "maelstrom-downscaling", prefix_nc: str = "downscaling_unet") -> None:
        super().__init__(datadir, app, fname_base=prefix_nc)

        self.status_ok = True

    def preprocess_data(self, ds_name: str, daytime: int =12, opt_norm: dict ={}):
        """
        Preprocess the data for feeding into the U-net, i.e. conversion to data arrays incl. z-score normalization
        :param ds_name: name of the dataset, i.e. one of the following "train", "val", "test"
        :param daytime: daytime in UTC for temporal slicing
        :param opt_norm: dictionary holding data for z-score normalization of data ("mu_in", "std_in", "mu_tar", "std_tar")
        :return: normalized data ready to be fed to U-net model
        """
        norm_dims_t = ["time"]  # normalization of 2m temperature for each grid point
        norm_dims_z = ["time", "lat", "lon"]  # 'global' normalization of surface elevation

        # slice the dataset
        dsf = self.data[ds_name].sel(time=dt.time(daytime))

        # retrieve and normalize input and target data
        if not opt_norm:
            t2m_in, t2m_in_mu, t2m_in_std = self.z_norm_data(dsf["t2m_in"], dims=norm_dims_t, return_stat=True)
            t2m_tar, t2m_tar_mu, t2m_tar_std = self.z_norm_data(dsf["t2m_tar"], dims=norm_dims_t, return_stat=True)
        else:
            t2m_in = self.z_norm_data(dsf["t2m_in"], mu=opt_norm["mu_in"], std=opt_norm["std_in"])
            t2m_tar = self.z_norm_data(dsf["t2m_tar"], mu=opt_norm["mu_tar"], std=opt_norm["std_tar"])

        z_in, z_tar = self.z_norm_data(dsf["z_in"], dims=norm_dims_z), self.z_norm_data(dsf["z_tar"], dims=norm_dims_z)

        in_data = xr.concat([t2m_in, z_in, z_tar], dim="variable")
        tar_data = xr.concat([t2m_tar, z_tar], dim="variable")

        # re-order data
        in_data = in_data.transpose("time",...,"variable")
        tar_data = tar_data.transpose("time",...,"variable")
        if not opt_norm:
            opt_norm = {"mu_in": t2m_in_mu, "std_in": t2m_in_std,
                        "mu_tar": t2m_tar_mu, "std_tar": t2m_tar_std}
            t_elapsed = t0 - timer()
            return in_data, tar_data, t_elapsed, opt_norm
        else:
            t_elapsed = t0 - timer()
            return in_data, tar_data, t_elapsed

    @staticmethod
    def z_norm_data(data: xr.Dataset, mu=None, std=None, dims=None, return_stat: bool =False):
        """
        Perform z-score normalization on the data
        :param data: the data as xarray-Dataset
        :param mu: the mean used for normalization (set to False if calculation from data is desired)
        :param std: the standard deviation used for normalization (set to False if calculation from data is desired)
        :param dims: list of dimension over which statistical quantities for normalization are calculated
        :param return_stat: flag if normalization statistics are returned
        :return: the normalized data
        """
        if mu is None and std is None:
            if not dims:
                dims = list(data.dims)
            mu = data.mean(dim=dims)
            std = data.std(dim=dims)

        data_out = (data - mu) / std

        if return_stat:
            return data_out, mu, std
        else:
            return data_out

# further auxiliary functions

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


# auxiliary function for colormap
def get_colormap_temp(levels=None):
    """
    Get a nice colormap for plotting topographic height
    :param levels: level boundaries
    :return cmap: colormap-object
    :return norm: normalization object corresponding to colormap and levels
    """
    bounds = np.asarray(levels)

    nbounds = len(bounds)
    col_obj = mpl.cm.PuOr_r(np.linspace(0., 1., nbounds))

    # create colormap and corresponding norm
    cmap = mpl.colors.ListedColormap(col_obj, name="temp" + "_map")
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    return cmap, norm, bounds


# for making plot nice
def decorate_plot(ax_plot, plot_xlabel=True, plot_ylabel=True):
    fs = 16
    if "login" in host:
        # add nice coast- and borderlines
        ax_plot.coastlines(linewidth=0.75)
        ax_plot.coastlines(linewidth=0.75)
        ax_plot.add_feature(cartopy.feature.BORDERS)

    # adjust extent and ticks as well as axis-label
    ax_plot.set_xticks(np.arange(0., 360. + 0.1, 5.))  # ,crs=projection_crs)
    ax_plot.set_yticks(np.arange(-90., 90. + 0.1, 5.))  # ,crs=projection_crs)

    ax_plot.set_extent([3.5, 17., 44.5, 55.])#, crs=prj_crs)
    ax_plot.minorticks_on()
    ax_plot.tick_params(axis="both", which="both", direction="out", labelsize=12)

    # some labels
    if plot_xlabel:
        ax_plot.set_xlabel("Longitude [°E]", fontsize=fs)
    if plot_ylabel:
        ax_plot.set_ylabel("Latitude[°N]", fontsize=fs)

    return ax_plot


# for creating plot
def create_plots(data1, data2, opt_plot={}):
    # get coordinate data
    try:
        time, lat, lon = data1["time"].values, data1["lat"].values, data1["lon"].values
        time_stamp = (pd.to_datetime(time)).strftime("%Y-%m-%d %H:00 UTC")
    except Exception as err:
        print("Failed to retrieve coordinates from data1")
        raise err
    # construct array for edges of grid points
    dy, dx = np.round((lat[1] - lat[0]), 2), np.round((lon[1] - lon[0]), 2)
    lat_e, lon_e = np.arange(lat[0]-dy/2, lat[-1]+dy, dy), np.arange(lon[0]-dx/2, lon[-1]+dx, dx)

    title1, title2 = provide_default(opt_plot, "title1", "input T2m"), provide_default(opt_plot, "title2", "target T2m")
    title1, title2 = "{0}, {1}".format(title1, time_stamp), "{0}, {1}".format(title2, time_stamp)
    levels = provide_default(opt_plot, "levels", np.arange(-5., 25., 1.))

    # get colormap
    cmap_temp, norm_temp, lvl = get_colormap_temp(levels)
    # create plot objects
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharex=True, sharey=True,
                                   subplot_kw={"projection": ccrs.PlateCarree()})

    # perform plotting
    temp1 = ax1.pcolormesh(lon_e, lat_e, np.squeeze(data1.values - 273.15), cmap=cmap_temp, norm=norm_temp)
    temp2 = ax2.pcolormesh(lon_e, lat_e, np.squeeze(data2.values - 273.15), cmap=cmap_temp, norm=norm_temp)

    ax1, ax2 = decorate_plot(ax1), decorate_plot(ax2, plot_ylabel=False)

    ax1.set_title(title1, size=14)
    ax2.set_title(title2, size=14)

    # add colorbar
    cax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
    cbar = fig.colorbar(temp2, cax=cax, orientation="vertical", ticks=lvl[1::2])
    cbar.ax.tick_params(labelsize=12)