# SPDX-FileCopyrightText: 2024 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Methods for creating plots.
"""

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-01-20"
__update__ = "2024-03-08"

# for processing data
import os
import logging
import numpy as np
import xarray as xr
import pandas as pd
# for plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs

# auxiliary variable for logger
logger_module_name = f"main_postprocess.{__name__}"
module_logger = logging.getLogger(logger_module_name)
module_name = os.path.basename(__file__).rstrip(".py")


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
def decorate_plot(ax_plot, plot_xlabel=True, plot_ylabel=True, extent=[2., 18., 42., 53.]):
    fs = 16
    # if "login" in host:
    # add nice coast- and borderlines
    ax_plot.coastlines(linewidth=0.75)
    ax_plot.coastlines(linewidth=0.75)
    ax_plot.add_feature(cartopy.feature.BORDERS)

    # adjust extent and ticks as well as axis-label
    ax_plot.set_xticks(np.arange(0., 360. + 0.1, 5.))  # ,crs=projection_crs)
    ax_plot.set_yticks(np.arange(-90., 90. + 0.1, 5.))  # ,crs=projection_crs)

    ax_plot.set_extent(extent)    # , crs=prj_crs)
    ax_plot.minorticks_on()
    ax_plot.tick_params(axis="both", which="both", direction="out", labelsize=fs)

    # some labels
    if plot_xlabel:
        ax_plot.set_xlabel("Longitude [°E]", fontsize=fs)
    if plot_ylabel:
        ax_plot.set_ylabel("Latitude[°N]", fontsize=fs)

    return ax_plot


# for creating plot
def create_mapplot(data1, data2, plt_fname, opt_plot={}):

    func_logger = logging.getLogger(f"postprocess.{module_name}.{create_mapplot.__name__}")

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

    title1, title2 = opt_plot.get("title1", "input T2m"), opt_plot.get("title2", "target T2m")
    title1, title2 = "{0}, {1}".format(title1, time_stamp), "{0}, {1}".format(title2, time_stamp)
    levels = opt_plot.get("levels", np.arange(-5., 25., 1.))

    # get colormap
    cmap_temp, norm_temp, lvl = get_colormap_temp(levels)
    # create plot objects
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), sharex=True, sharey=True,
                                   subplot_kw={"projection": ccrs.PlateCarree()})

    # perform plotting
    _ = ax1.pcolormesh(lon_e, lat_e, np.squeeze(data1.values), cmap=cmap_temp, norm=norm_temp)
    temp2 = ax2.pcolormesh(lon_e, lat_e, np.squeeze(data2.values), cmap=cmap_temp, norm=norm_temp)

    ax1, ax2 = decorate_plot(ax1), decorate_plot(ax2, plot_ylabel=False)

    ax1.set_title(title1, size=14)
    ax2.set_title(title2, size=14)

    # add colorbar
    cax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
    cbar = fig.colorbar(temp2, cax=cax, orientation="vertical", ticks=lvl[1::2])
    cbar.ax.tick_params(labelsize=12)

    # save plot and close figure
    plt_fname = plt_fname + ".png" if not plt_fname.endswith(".png") else plt_fname
    func_logger.info(f"Save plot in file '{plt_fname}'")
    fig.savefig(plt_fname, bbox_inches="tight")
    plt.close(fig)

    
def create_map_score(score, plt_fname, **kwargs):

    func_logger = logging.getLogger(f"postprocess.{module_name}.{create_map_score.__name__}")

    # get keywor arguments
    dims = kwargs.get("dims", ["lat", "lon"])
    title = kwargs.get("title", "Score")
    levels = kwargs.get("levels", np.arange(-5., 5., 0.5))
    # auxiliary variables
    lvl = np.asarray(levels)
    nbounds = len(lvl)
    cmap = kwargs.get("cmap", mpl.cm.PuOr_r(np.linspace(0., 1., nbounds)))
    fs = kwargs.get("fs", 16)
    projection = kwargs.get("projection", ccrs.RotatedPole(pole_longitude=-162.0, pole_latitude=39.25))
    extent = kwargs.get("extent", None)
    
    decorate_dict = {}
    if extent:
        decorate_dict["extent"] = extent
    
    # get coordinate data
    try:
        lat, lon = score[dims[0]].values, score[dims[1]].values
    except Exception as err:
        print("Failed to retrieve coordinates from score-data")
        raise err
    # construct array for edges of grid points
    dy, dx = np.round((lat[1] - lat[0]), 4), np.round((lon[1] - lon[0]), 4)
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

    ax = decorate_plot(ax, **decorate_dict)

    ax.set_title(title, size=fs)

    # add colorbar
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(plt1, cax=cax, orientation="vertical", ticks=lvl[1::2])
    cbar.ax.tick_params(labelsize=fs-2)

    # save plot and close figure
    plt_fname = plt_fname + ".png" if not plt_fname.endswith(".png") else plt_fname
    func_logger.info(f"Save plot in file '{plt_fname}'")
    fig.savefig(plt_fname, bbox_inches="tight")
    plt.close(fig)


def create_line_plot(data: xr.DataArray, data_std: xr.DataArray, model_name: str, metric: dict,
                     plt_fname: str, x_coord: str = "hour", **kwargs):

    func_logger = logging.getLogger(f"postprocess.{module_name}.{create_line_plot.__name__}")

    # get some plot parameters
    linestyle = kwargs.get("linestyle", "k-")
    err_col = kwargs.get("error_color", "blue")
    val_range = kwargs.get("value_range", (0., 4.))
    fs = kwargs.get("fs", 16)
    ref_line = kwargs.get("ref_line", None)
    ref_linestyle = kwargs.get("ref_linestyle", "k--")
    
    fig, (ax) = plt.subplots(1, 1)
    ax.plot(data[x_coord].values, data.values, linestyle, label=model_name)
    ax.fill_between(data[x_coord].values, data.values-data_std.values, data.values+data_std.values, facecolor=err_col,
                    alpha=0.2)
    if ref_line is not None:
        nval = np.shape(data[x_coord].values)[0]
        ax.plot(data[x_coord].values, np.full(nval, ref_line), ref_linestyle)
    ax.set_ylim(*val_range)
    # label axis
    ax.set_xlabel("daytime [UTC]", fontsize=fs)
    metric_name, metric_unit = list(metric.keys())[0], list(metric.values())[0]
    ax.set_ylabel(f"{metric_name} T2m [{metric_unit}]", fontsize=fs)
    ax.tick_params(axis="both", which="both", direction="out", labelsize=fs-2)

    # save plot and close figure
    plt_fname = plt_fname + ".png" if not plt_fname.endswith(".png") else plt_fname
    func_logger.info(f"Save plot in file '{plt_fname}'")
    fig.savefig(plt_fname, bbox_inches="tight")
    plt.tight_layout()
    fig.savefig(plt_fname)
    plt.close(fig)


# write the create_box_plot function
def create_box_plot(data, plt_fname: str, **plt_kwargs):
    """
    Create box plot of feature importance scores
    :param feature_scores: Feature importance scores with predictors as firstdimension and time as second dimension
    :param plt_fname: File name of plot
    """    
    func_logger = logging.getLogger(f"postprocess.{module_name}.{create_box_plot.__name__}")

    # get some plot parameters
    val_range = plt_kwargs.get("value_range", [None])
    widths = plt_kwargs.get("widths", None)
    colors = plt_kwargs.get("colors", None)
    fs = plt_kwargs.get("fs", 16)
    ref_line = plt_kwargs.get("ref_line", 1.)
    ref_linestyle = plt_kwargs.get("ref_linestyle", "k-")
    title = plt_kwargs.get("title", "")
    ylabel = plt_kwargs.get("ylabel", "")
    xlabel = plt_kwargs.get("xlabel", "")
    yticks = plt_kwargs.get("yticks", None)
    labels = plt_kwargs.get("labels", None)
    
    # create box whiskers plot with matplotlib
    fig, ax = plt.subplots(figsize=(12, 8))

    bp = plt.boxplot(data, widths=widths, labels=labels, patch_artist=True)
    
    # modify fliers
    fliers = bp['fliers'] 
    for i in range(len(fliers)): # iterate through the Line2D objects for the fliers for each boxplot
        box = fliers[i] # this accesses the x and y vectors for the fliers for each box 
        box.set_data([[box.get_xdata()[0]],[np.max(box.get_ydata())]])
        
    if ref_line is not None:
        nval = len(fliers)
        ax.plot(np.array(range(0, nval+1)) + 0.5, np.full(nval+1, ref_line), ref_linestyle)
        
    if colors is None:
        pass
    else:
        if isinstance(colors, str): colors = len(bp["boxes"])*[colors]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
    
    ax.set_ylim(*val_range)
    ax.set_yticks(yticks)    
    
    ax.set_title(title, fontsize=fs + 2)
    ax.set_ylabel(ylabel, fontsize=fs, labelpad=8)
    ax.set_xlabel(xlabel, fontsize=fs, labelpad=8)
    ax.tick_params(axis="both", which="both", direction="out", labelsize=fs-2)
    ax.yaxis.grid(True)

    # save plot
    plt.tight_layout()
    plt.savefig(plt_fname + ".png" if not plt_fname.endswith(".png") else plt_fname)
    plt.close(fig)

    func_logger.info(f"Feature importance scores saved to {plt_fname}.")
    
    return True


def create_ps_plot(ds_ps: xr.Dataset, var_info: dict, labels: List[str], plt_fname: str, x_coord: str = "wavenumber", **kwargs):
    """
    Plots power spectrum.
    :param ds_ps: Dataset providing power spectrum of experiments as DataArrays
    :param var_info: Dictionary providing name of variable and unit for which spectrum/spectra is/are poltted
    :param labels: List of labels for experiments
    :param plt_fname: File name of plot
    :param x_coord: Name of coordinate along which spectrum is plotted
    :param kwargs: Keyword arguments for plotting
    """
    # auxiliary variables
    exps = list(ds_ps.data_vars)
    nexps = len(exps)
    assert nexps == len(labels), "Number of labels must match number of experiments"

    # get some plot parameters
    linestyle = kwargs.get("linestyle", "k-")
    lw = kwargs.get("linewidth", 2.)
    cols = kwargs.get("colors", nexps*["blue"])
    fs = kwargs.get("fs", 16)
    
    fig, (ax) = plt.subplots(1, 1)#, figsize=(12, 8))
    for i, exp in enumerate(exps):
        da = ds_ps[exp]
        ax.plot(da[x_coord].values, da.values, linestyle, label=labels[i], lw=lw, c=cols[i])

    # set axis limits
    ax.set_yscale("log")
    ax.set_title(f"")
    # label axis
    ax.set_xlabel("wavenumber", fontsize=fs)
    var_name, spectrum_unit = list(var_info.keys())[0], list(var_info.values())[0]
    ax.set_ylabel(f"Spectral power {var_name} [{spectrum_unit}]", fontsize=fs)
    ax.tick_params(axis="both", which="both", direction="out", labelsize=fs-2)
    ax.legend(fontsize=fs-2)
    
    # save plot and close figure
    plt_fname = plt_fname + ".png" if not plt_fname.endswith(".png") else plt_fname
    print(f"Save plot in file '{plt_fname}'")
    plt.tight_layout()
    fig.savefig(plt_fname)
    plt.close(fig)
