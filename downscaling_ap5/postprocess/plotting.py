__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-01-20"
__update__ = "2022-01-22"

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
sys.path.append("../utils")
from other_utils import provide_default
#from input_data_class import InputDataClass


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
    #if "login" in host:
        # add nice coast- and borderlines
    ax_plot.coastlines(linewidth=0.75)
    ax_plot.coastlines(linewidth=0.75)
    ax_plot.add_feature(cartopy.feature.BORDERS)

    # adjust extent and ticks as well as axis-label
    ax_plot.set_xticks(np.arange(0., 360. + 0.1, 5.))  # ,crs=projection_crs)
    ax_plot.set_yticks(np.arange(-90., 90. + 0.1, 5.))  # ,crs=projection_crs)

    ax_plot.set_extent([3.5, 16.5, 44.5, 54.])#, crs=prj_crs)
    ax_plot.minorticks_on()
    ax_plot.tick_params(axis="both", which="both", direction="out", labelsize=fs)

    # some labels
    if plot_xlabel:
        ax_plot.set_xlabel("Longitude [°E]", fontsize=fs)
    if plot_ylabel:
        ax_plot.set_ylabel("Latitude[°N]", fontsize=fs)

    return ax_plot


# for creating plot
def create_mapplot(data1, data2, plt_fname ,opt_plot={}):
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
    
    fig.savefig(plt_fname+".png", bbox_inches="tight")
    plt.close(fig)

    
def create_map_score(score, plt_fname, **kwargs):
    # get keywor arguments
    score_dims = kwargs.get("score_dims", ["lat", "lon"])
    title = kwargs.get("title", "Score")
    levels = kwargs.get("levels", np.arange(-5., 5., 0.5))
    # auxiliary variables
    lvl = np.asarray(levels)
    nbounds = len(lvl)
    cmap = kwargs.get("cmap", mpl.cm.PuOr_r(np.linspace(0., 1., nbounds)))
    fs = kwargs.get("fs", 16)
    projection = kwargs.get("projection", ccrs.PlateCarree())
    
    # get coordinate data
    try:
        lat, lon = score[score_dims[0]].values, score[score_dims[1]].values
    except Exception as err:
        print("Failed to retrieve coordinates from score-data")
        raise err
    # construct array for edges of grid points
    dy, dx = np.round((lat[1] - lat[0]), 3), np.round((lon[1] - lon[0]), 3)
    lat_e, lon_e = np.arange(lat[0]-dy/2, lat[-1]+dy, dy), np.arange(lon[0]-dx/2, lon[-1]+dx, dx)

    # get colormap
    # create colormap and corresponding norm
    cmap_obj = mpl.colors.ListedColormap(cmap, name="temp" + "_map")
    norm = mpl.colors.BoundaryNorm(lvl, cmap_obj.N)
    # create plot objects
    fig, ax = plt.subplots(1, 1, figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()})

    # perform plotting
    plt1 = ax.pcolormesh(lon_e, lat_e, np.squeeze(score.values), cmap=cmap_obj, norm=norm, 
                         transform=projection)

    ax = decorate_plot(ax)

    ax.set_title(title, size=fs)

    # add colorbar
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(plt1, cax=cax, orientation="vertical", ticks=lvl[1::2])
    cbar.ax.tick_params(labelsize=fs-2)
    
    fig.savefig(plt_fname+".png", bbox_inches="tight")
    plt.close(fig)

def create_line_plot(data: xr.DataArray, data_std: xr.DataArray, model_name: str, metric: dict,
                     filename: str, x_coord: str = "hour", **kwargs):
    
    # get some plot parameters
    linestyle = kwargs.get("linestyle", "k-")
    err_col = kwargs.get("error_color", "blue")
    val_range = kwargs.get("value_range", (0., 4.))
    fs = kwargs.get("fs", 16)
    ref_line = kwargs.get("ref_line", None)
    ref_linestyle = kwargs.get("ref_linestyle", "k--")
    
    fig, (ax) = plt.subplots(1,1)
    ax.plot(data[x_coord].values, data.values, linestyle, label=model_name)
    ax.fill_between(data[x_coord].values, data.values-data_std.values, data.values+data_std.values, facecolor=err_col, alpha=0.2)
    if ref_line is not None:
        nval = np.shape(data[x_coord].values)[0]
        ax.plot(data[x_coord].values, np.full(nval, ref_line), ref_linestyle)
    ax.set_ylim(*val_range)
    # label axis
    ax.set_xlabel("daytime [UTC]", fontsize=fs)
    metric_name, metric_unit = list(metric.keys())[0], list(metric.values())[0]
    ax.set_ylabel(f"{metric_name} T2m [{metric_unit}]", fontsize=fs)
    ax.tick_params(axis="both", which="both", direction="out", labelsize=fs-2)

    # save plot to file
    plt.tight_layout()
    fig.savefig(filename)