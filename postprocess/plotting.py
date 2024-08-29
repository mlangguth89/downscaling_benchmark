# SPDX-FileCopyrightText: 2024 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Methods for creating plots.
"""

__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-01-20"
__update__ = "2024-04-16"

# for processing data
import os
from typing import List
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
def get_cmap_norm(levels, cb_name: str = "PuOr_r", cb_range= (0., 1.)):
    """
    Get the colormap and norm-object for given levels and a given colorbar-name
    :param levels: level boundaries
    :param cb_name: name of colorbar 
    :return cmap: colormap-object
    :return norm: normalization object corresponding to colormap and levels
    """
    bounds = np.asarray(levels)
    nbounds = len(bounds)
    
    col_obj = plt.get_cmap(cb_name)
    col_obj = col_obj(np.linspace(*cb_range, nbounds)) 

    # create colormap and corresponding norm
    cmap = mpl.colors.ListedColormap(col_obj)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    return cmap, norm


def get_season_t2m_levels(date, dx: int = 2):
    """
    Get levels for plotting temperature data in different seasons
    :param date: date for which levels are calculated
    :param dx: step size between levels
    :return lvl: levels for plotting temperature data
    """
    month = int(pd.to_datetime(date).strftime("%m"))

    # season_inds [0, 1, 2, 3] correspond to ["DJF", "MAM", "JJA", "SON"] 
    season_ind = month%12 // 3 
    
    if season_ind == 0:
        offset = 0
    elif season_ind == 2:
        offset = 20
    else:
        offset = 10
        
    lvl = np.arange(-22. + offset, 22.1 + offset, dx)
    
    return lvl

# for making plot nice
def decorate_plot(ax_plot, plot_xlabel=True, plot_ylabel=True, extent=[2., 18., 42., 53.], fs = 16):
    """
    Decorate plot with coastlines, borders and define extent, ticks and axis-labels
    :param ax_plot: plot object
    :param plot_xlabel: boolean for apparance x-axis label
    :param plot_ylabel: boolean for apparance y-axis label
    :param extent: extent of plot domain [west, east, south, north]
    :param fs: font size of labels
    """
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
def plot_comparison_maps(ds: xr.Dataset, plt_fname: str, **kwargs):
    """
    Plots two variables from a dataset and their difference next to each other, i.e. resulting in a 1x3 plot.
    :param ds: dataset eithher providing two variables only for comparison
               or variables provided with vars2plt-keyword argument
    :param plt_fname: path to output filename
    :param kwargs: valid keyword arguments are
                  - vars2plt: list of two variables to compare
                  - titles: list of plot titles for both variables
                  - proj_data: cartopy projection-object for the data
                  - proj_plot: cartopy projection-object for the plotted domain
                  - dims: name of spatial coordinates, e.g. ["lat", "lon"]
                  - extent: geospatial extension of plot domain in degree
                  - fs: basic font size of labels
                  - figsize: figure size in inch
                  - levels: levels for plotting both variables
                  - cmap_name: colormap name used for plotting both variables
                  - levels_diff: levels for the difference plot 
                  - cmap_name_diff: colormap name for the difference plots
                  - cbar_shrink: shrink-factor for the vertically aligned colorbar
                  - further valid arguments of ax.pcolormesh           
    """
    func_logger = logging.getLogger(f"postprocess.{module_name}.{plot_comparison_maps.__name__}")

    # check and get keyword arguments
    vars2plt = kwargs.pop("vars2plt", None)
    titles = kwargs.pop("titles", None)
    proj_data = kwargs.pop("proj_data", ccrs.RotatedPole(pole_longitude=-162.0, pole_latitude=39.25))
                      
    proj_plot = kwargs.pop("proj_plot", ccrs.PlateCarree())
    dims = kwargs.pop("dims", ["rlat", "rlon"])
    extent = kwargs.pop("extent", [3., 16.5, 43., 51.5])
    fs = kwargs.pop("fs", 16)
    figsize = kwargs.pop("figsize", (24, 6))
    # get levels and colorbars for 'normal' data plots and difference plots
    levels = kwargs.pop("levels", np.arange(-22., 42.1, 2))
    cmap_name = kwargs.pop("cmap_name", "jet")
    levels_diff = kwargs.pop("levels_diff", np.arange(-5.25, 5.01, 0.5))
    cmap_name_diff = kwargs.pop("cmap_name_diff", "PuOr_r")
    cbar_shrink = .8
    
    # auxiliary variables
    lvl, lvl_diff = np.asarray(levels), np.asarray(levels_diff)
    
    if vars2plt:
        vars2plt = list(vars2plt)
    else:
        vars2plt = list(ds.data_vars)
        
    nplots = len(vars2plt)
    
    assert nplots == 2, f"Number of variables to plot must be 2, but is {nplots}."
        
    if titles:
        assert len(titles) == nplots, f"Number of variables to plot ({nplots}) " + \
                                      f"must match number of titles ({len(titles)})."
    else:
        titles = [var.replace("_", " ").upper() for var in vars2plt]
    
    # get coordinate data
    try:
        var_now = ds[vars2plt[0]]
        lat, lon = var_now[dims[0]].values, var_now[dims[1]].values
    except Exception as err:
        func_logger.debug(f"Failed to retrieve coordinates from {vars2plt[0]}")
        raise err
    # construct array for edges of grid points
    dy, dx = np.round((lat[1] - lat[0]), 4), np.round((lon[1] - lon[0]), 4)
    lat_e, lon_e = np.arange(lat[0]-dy/2, lat[-1]+dy, dy), np.arange(lon[0]-dx/2, lon[-1]+dx, dx)

    # get colormap
    cmap, norm = get_cmap_norm(levels, cmap_name)
    cmap_diff, norm_diff = get_cmap_norm(levels_diff, cmap_name_diff)
    
    # create plot objects
    fig, axs = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True,
                            subplot_kw={"projection": proj_plot})
        
    # perform plotting
    for i, ax in enumerate(axs):
        if i < nplots:
            data = np.squeeze(ds[vars2plt[i]])
            
            plt_data = ax.pcolormesh(lon_e, lat_e, data.values, cmap=cmap, norm=norm, transform=proj_data,
                                    **kwargs)
            ax.set_title(titles[i], size=fs)                
        else:
            diff = np.squeeze(ds[vars2plt[1]] - ds[vars2plt[0]])
            plt_diff = ax.pcolormesh(lon_e, lat_e, diff.values, cmap=cmap_diff, norm=norm_diff,
                                     transform=proj_data, **kwargs)
            ax.set_title("Difference", size=fs)
        
        # custom plot apparance
        ax = decorate_plot(ax, plot_ylabel= i == 0, extent=extent, fs=fs)
    
    # add colorbars
    cbar = fig.colorbar(plt_data, ax=axs[0:2], orientation="vertical", shrink=cbar_shrink,
                        pad=.02, ticks=lvl[1::2], fraction=0.02)
    cbar.ax.tick_params(labelsize=fs-2)
    
    cbar_diff = fig.colorbar(plt_diff, ax=axs[-1], orientation="vertical", shrink=1.5,
                             pad=.04, ticks=lvl_diff[1::2], fraction=0.02)
    cbar_diff.ax.tick_params(labelsize=fs-2)

    # save plot and close figure
    plt_fname = plt_fname + ".png" if not plt_fname.endswith(".png") else plt_fname
    func_logger.info(f"Save comparison plot in file '{plt_fname}'")
    fig.savefig(plt_fname, bbox_inches="tight", dpi=300)
    plt.close(fig)

    
def plot_score_map(score, plt_fname, **kwargs):
    """
    Plots a score on a map.
    :param score: DataArray containing the score
    :param plt_fname: path to output filename
    :param kwargs: valid keyword arguments are
                  - title: title of the plot
                  - proj_data: cartopy projection-object for the data
                  - proj_plot: cartopy projection-object for the plotted domain
                  - dims: name of spatial coordinates, e.g. ["lat", "lon"]
                  - extent: geospatial extension of plot domain in degree
                  - fs: basic font size of labels
                  - figsize: figure size in inch
                  - levels: levels for plotting the score
                  - cmap_name: colormap name used for plotting the score
                  - further valid arguments of ax.pcolormesh
    """
    func_logger = logging.getLogger(f"postprocess.{module_name}.{plot_score_map.__name__}")

    # get keyword arguments
    title = kwargs.pop("title", "Score")
    proj_data = kwargs.pop("proj_data", ccrs.RotatedPole(pole_longitude=-162.0, pole_latitude=39.25))
    proj_plot = kwargs.pop("proj_plot", ccrs.PlateCarree())
    dims = kwargs.pop("dims", ["rlat", "rlon"])
    extent = kwargs.pop("extent", [3., 16.5, 43., 51.5])
    fs = kwargs.pop("fs", 16)
    figsize = kwargs.pop("figsize", (12, 8))
    # get levels and colorbars 
    levels = kwargs.pop("levels", np.arange(-5.25, 5.01, 0.5))
    cmap_name = kwargs.pop("cmap_name", "PuOr_r")

    # auxiliary variables
    lvl = np.asarray(levels)
    
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
    cmap, norm = get_cmap_norm(levels, cmap_name)
    # create plot objects
    fig, ax = plt.subplots(1, 1, figsize=figsize, subplot_kw={"projection": proj_plot})

    # perform plotting
    plt1 = ax.pcolormesh(lon_e, lat_e, np.squeeze(score.values), cmap=cmap, norm=norm, 
                         transform=proj_data)

    ax = decorate_plot(ax, extent=extent, fs=fs)

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


def plot_metric_line(data: xr.DataArray, data_up: xr.DataArray, data_down: xr.DataArray, model_name: str, metric: dict,
                     plt_fname: str, varname: str = "T2m", x_coord: str = "hour", **kwargs):
    """
    Create line plots of 2D-metric data (e.g. metric plotted against time) 
    :param data: DataArray containing the mean values
    :param data_up: DataArray containing the upper error bounds
    :param data_down: DataArray containing the lower error bounds
    :param model_name: Name of model
    :param metric: Dictionary containing metric name and unit 
    :param plt_fname: File name of plot
    :param varname: Name of variable that was evaluated
    :param x_coord: Name of coordinate along which metric is plotted
    :param kwargs: Keyword arguments for plotting
                   Valid keys are:
                    - "linestyle": linestyle of plot, default: "k-"
                    - "error_color": color of error bounds, default: "blue"
                    - "value_range": range of y-axis, default: (0., 4.)
                    - "fs": font size of labels, default: 16
                    - "ref_line": reference line to be plotted, default: None
                    - "ref_linestyle": linestyle of reference line, default: "k--"
                    - other valid arguments of ax.plot
    """
    func_logger = logging.getLogger(f"postprocess.{module_name}.{plot_metric_line.__name__}")

    # get some plot parameters
    linestyle = kwargs.pop("linestyle", "k-")
    err_col = kwargs.pop("error_color", "blue")
    val_range = kwargs.pop("value_range", (0., 4.))
    fs = kwargs.pop("fs", 16)
    ref_line = kwargs.pop("ref_line", None)
    ref_linestyle = kwargs.pop("ref_linestyle", "k--")
    
    fig, (ax) = plt.subplots(1, 1)
    ax.plot(data[x_coord].values, data.values, linestyle, label=model_name, **kwargs)
    ax.fill_between(data[x_coord].values, data_down.values, data_up.values, facecolor=err_col,
                    alpha=0.2)
    if ref_line is not None:
        nval = np.shape(data[x_coord].values)[0]
        ax.plot(data[x_coord].values, np.full(nval, ref_line), ref_linestyle)
    ax.set_ylim(*val_range)
    # label axis
    ax.set_xlabel("daytime [UTC]", fontsize=fs)
    metric_name, metric_unit = list(metric.keys())[0], list(metric.values())[0]
    ax.set_ylabel(f"{metric_name} {varname} [{metric_unit}]", fontsize=fs)
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
    :param plt_kwargs: Keyword arguments for plotting
                       Valid keys are:
                        - "value_range": range of y-axis, default: [None]
                        - "widths": width of boxes, default: None
                        - "colors": color of boxes, default: None
                        - "fs": font size of labels, default: 16
                        - "ref_line": reference line to be plotted, default: 1.
                        - "ref_linestyle": linestyle of reference line, default: "k-"
                        - "title": title of plot, default: ""
                        - "ylabel": label of y-axis, default: ""
                        - "xlabel": label of x-axis, default: ""
                        - "yticks": ticks of y-axis, default: None
                        - "labels": labels of boxes, default: None
                        - other valid arguments of plt.boxplot
    """    
    func_logger = logging.getLogger(f"postpess.{module_name}.{create_box_plot.__name__}")

    # get parametrs that should not be parsed to the boxplot-method 
    figsize = plt_kwargs.pop("figsize", (12, 8))
    val_range = plt_kwargs.pop("value_range", [None])
    widths = plt_kwargs.pop("widths", .3)
    colors = plt_kwargs.pop("colors","black")
    fs = plt_kwargs.pop("fs", 16)
    ref_line = plt_kwargs.pop("ref_line", 1.)
    ref_linestyle = plt_kwargs.pop("ref_linestyle", "-")
    title = plt_kwargs.pop("title", "")
    ylabel = plt_kwargs.pop("ylabel", "")
    xlabel = plt_kwargs.pop("xlabel", "")
    yticks = plt_kwargs.pop("yticks", None)
    labels = plt_kwargs.pop("labels", None)
    
    # create box whiskers plot with matplotlib
    fig, ax = plt.subplots(figsize=figsize)

    bp = plt.boxplot(data, widths=widths, labels=labels, patch_artist=True, **plt_kwargs)
    
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


def plot_power_spectra(ds_ps: xr.Dataset, var_info: dict, labels: List[str], plt_fname: str, x_coord: str = "wavenumber", **kwargs):
    """
    Plots power spectrum.
    :param ds_ps: Dataset providing power spectrum of experiments as DataArrays
    :param var_info: Dictionary providing name of variable and unit for which spectrum/spectra is/are poltted
    :param labels: List of labels for experiments
    :param plt_fname: File name of plot
    :param x_coord: Name of coordinate along which spectrum is plotted
    :param kwargs: Keyword arguments for plotting
                   Valid keys are:
                     - "linestyle": linestyle of plot, default: "-"
                     - "linewidth": linewidth of plot, default: 2.
                     - "colors": list of colors for each experiment, default: nexps*["blue"]
                     - "fs": font size of labels, default: 16
                     - other valid arguments of ax.plot
    """
    # auxiliary variables
    exps = list(ds_ps.data_vars)
    nexps = len(exps)
    assert nexps == len(labels), "Number of labels must match number of experiments"

    # get some plot parameters
    linestyle = kwargs.pop("linestyle", "-")
    lw = kwargs.pop("linewidth", 2.)
    cols = kwargs.pop("colors", nexps*["blue"])
    fs = kwargs.pop("fs", 16)
    
    fig, (ax) = plt.subplots(1, 1)#, figsize=(12, 8))
    for i, exp in enumerate(exps):
        da = ds_ps[exp]
        ax.plot(da[x_coord].values, da.values, linestyle, label=labels[i], lw=lw, c=cols[i], **kwargs)

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

def plot_cond_quantile(quantile_panel: xr.DataArray, data_marginal: xr.DataArray, plt_fname: str, opt: dict = {}):
    """
    Creates conditional quantile plot
    :param quantile_panel: quantile panel created by calculate_cond_quantiles
    :param data_marginal: data array for which histogram will be plotted
    :param plt_fname: name of the plot-file to be created
    :param opt: options to customize the plot
                Valid keys are:
                - "figsize": tuple with dimensions of figure, default: (12, 6)
                - "fs_title": font size of title, default: 16)
                - "fs_axis_label": font size of axis labels, default: fs_title-2
                - "plt_title": title of plot, default: ""
    :return:
    """
    func_logger = logging.getLogger(f"postprocess.{module_name}.{plot_cond_quantile.__name__}") 

    if not isinstance(quantile_panel, xr.DataArray):
        raise ValueError("quantile_panel must be a DataArray, but is a {0}".format(type(quantile_panel)))

    if not isinstance(data_marginal, xr.DataArray):
        raise ValueError("data_marginal must be a DataArray, but is a {0}".format(type(data_marginal)))

    if list(quantile_panel.coords) != ["bin_center", "quantile"]:
        raise ValueError("The coordinates of quantile_panel must be ['bin_center', 'quantile']. Use calculate_cond_quantiles to calculate them.")

    if opt is None:
        opt = {}

    func_logger.info(f"Start creating conditional quantile plot in file '{plt_fname}'")

    bins_c = quantile_panel["bin_center"]
    bin_width = bins_c[1] - bins_c[0]
    bins = np.arange(bins_c[0]-bin_width/2., bins_c[-1]+1.5*bin_width/2, bin_width)
    quantiles = quantile_panel["quantile"]
    nquantiles = len(quantiles)
    if nquantiles%2 != 1:
        raise ValueError(f"Number of quantiles must be odd, but is {nquantiles}.")

    # auxiliary functions
    def get_ls_mirrored(n, ls_base=("--", ":")):

        nls_base = len(ls_base)
        lss = []
        for ilw in np.arange(n):
            if ilw < nls_base:
                lss.append(ls_base[ilw])
            else:
                lss.append("-")

        lss = lss + ["-"] + lss[::-1]

        return lss

    ls_all = get_ls_mirrored(int(nquantiles/2))
    lw_all = list(np.full(nquantiles, 2.))
    lw_all[int(nquantiles/2)] = 1.5

    # start plotting
    figsize = opt.get("figsize", (12, 6))
    fs_title = opt.get("fs_axis_title", 16)
    fs_label = opt.get("fs_axis_label", fs_title-2)
    plt_title = opt.get("plt_title", "")
    fig, ax = plt.subplots(figsize=figsize)

    # plot reference line
    ax.plot(bins_c, bins_c, color='k', label='reference 1:1', linewidth=1.)
    # plot conditional quantiles
    for iq in np.arange(nquantiles):
        ax.plot(bins_c, quantile_panel.isel(quantile=iq), ls=ls_all[iq], color="k", lw=lw_all[iq],
                label="{0:d}th quantile".format(int(quantiles[iq]*100.)))
    # plot histogram of marginal distribution
    ax2 = ax.twinx()
    xr.plot.hist(data_marginal, ax=ax2, bins=bins, color="k", alpha=0.3)
    ax2.set_yscale("log")
    
    qp_attrs = dict(quantile_panel.attrs)

    xlabel = "{0} [{1}]".format(qp_attrs.get("cond_varname", "conditiong variable"),
                                qp_attrs.get("unit", "unknown"))
    ylabel = "{0} [{1}]".format(qp_attrs.get("tar_varname", "target variable"),
                                qp_attrs.get("unit", "unknown"))

    ax.set_ylabel(ylabel, fontsize=fs_title)
    ax2.set_ylabel("counts", fontsize=fs_title)
    ax.set_xlabel(xlabel, fontsize=fs_title)
    # ensure that histogram extends to the lower half of the plot
    y2_max_power = int(np.log10(ax2.get_ylim()[1]))
    ax2.set(ylim=(1.e00, np.power(10, y2_max_power*4)), yticks=np.logspace(0, y2_max_power+1, y2_max_power+2)) 
    ax2.set_title(plt_title)

    ax.tick_params(axis="both", labelsize=fs_label)
    ax2.tick_params(axis="both", labelsize=fs_label)

    fig.savefig(plt_fname)
    plt.close("all")

