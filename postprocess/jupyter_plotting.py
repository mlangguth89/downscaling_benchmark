# SPDX-FileCopyrightText: 2024 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Methods for creating plots.
"""

__author__ = "Ankit Langguth"
__email__ = "a.patnala@fz-juelich.de"
__date__ = "2024-04-26"
__update__ = "2024-04-26"

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
from plotting import get_cmap_norm
import cartopy.feature as cfeature

# auxiliary variable for logger
logger_module_name = f"main_postprocess.{__name__}"
module_logger = logging.getLogger(logger_module_name)
module_name = os.path.basename(__file__).rstrip(".py")

def plot_skills(data, plt_fname, labels=["U-Net (Sha)", "WGAN (Sha)", "DeepRU", "SwinIR"], metric="RMSE",**kwargs):

    fs = 16
    # create figure
    fig, ax = plt.subplots(1, 1)
    # create box-plot
    bp = ax.boxplot(data.T, labels=labels, patch_artist=True)
    # configure plot
    minimum = np.min(data)*0.95
    maximum = np.max(data)*1.05
    ax.set_ylim(minimum, maximum)
    #all external decorartive arguments
    title = kwargs.pop("title","")
    colors = kwargs.pop("colors",['pink', 'lightblue', 'lightgreen','blue'])
    show = kwargs.pop("show",True)
    savefig = kwargs.pop("savefig",False)

    ax.set_title(title)
    ax.set_ylabel(f"Skill {metric}", fontsize=fs)
    ax.tick_params(axis="both", which="both", direction="out", labelsize=fs-2)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2.)

    plt.rcParams['text.usetex'] = True
    #ax.text(0.7, 0.05, r"$\overline{RMSE}_{ref}$="+f"{rmse_ref.mean():.2f}K",
    #    transform=ax.transAxes,
    #    color='k', fontsize=fs-2)
    if show:
        plt.show()
    if savefig:
        fig.savefig(plt_fname, bbox_inches="tight")
    plt.close(fig)


def create_lines_plot(data, data_std, model_names: str, metric: dict,
                      plt_fname: str,  **kwargs):

    # get some plot parameters
    if data_std is not None:
        assert(len(data) == len(data_std))
    linestyle = kwargs.get("linestyle", ["k-", "b-","o-","r-"])
    err_col = kwargs.get("error_color", ["grey", "blue","green","red"])
    val_range = kwargs.get("value_range", (0., 3.))
    fs = kwargs.get("fs", 16)
    ref_line = kwargs.get("ref_line", None)
    ref_linestyle = kwargs.get("ref_linestyle", "k--")
    show = kwargs.pop("show",True)
    savefig = kwargs.pop("savefig",False)
    
    fig, (ax) = plt.subplots(1, 1)
    for i, exp in enumerate(data):
        ax.plot(np.arange(24), data[i], linestyle[i],
                label=model_names[i])
        if data_std is not None:
            ax.fill_between(np.arange(24), data[i]-data_std[i],
                            data[i]+data_std[i], facecolor=err_col[i],
                            alpha=0.2)
    ax.set_ylim(*val_range)
    # label axis
    ax.set_xlabel("daytime [UTC]", fontsize=fs)
    metric_name, metric_unit = list(metric.keys())[0], list(metric.values())[0]
    ax.set_ylabel(f"{metric_name} [{metric_unit}]", fontsize=fs)
    ax.tick_params(axis="both", which="both", direction="out", labelsize=fs-2)
    ax.legend(fontsize=fs-2, loc="upper right")
    
    if show:
        plt.show()

    if savefig:
        # save plot and close figure
        plt_fname = plt_fname + ".png" if not plt_fname.endswith(".png") else plt_fname
        print(f"Save plot in file '{plt_fname}'")
        plt.tight_layout()

        fig.savefig(plt_fname)
    plt.close(fig)

def power_spectra_plot(ds_ps: list[xr.DataArray], var_info: dict, labels: list[str], plt_fname: str, x_coord: str = "wavenumber",**kwargs):

    # get some plot parameters
    linestyle = kwargs.pop("linestyle", "-")
    lw = kwargs.pop("linewidth", 2.)
    cols = kwargs.pop("colors", ["blue","green","orange","red"])
    fs = kwargs.pop("fs", 16)
    show = kwargs.pop("show",True)
    savefig = kwargs.pop("savefig",False)

    fig, (ax) = plt.subplots(1, 1)#, figsize=(12, 8))
    for i, exp in enumerate(ds_ps):
        da = ds_ps[i]
        ax.plot(da[x_coord].values, da.values, linestyle, label=labels[i], lw=lw, color=cols[i], **kwargs)
    ax.set_yscale("log")
    ax.set_title(f"")
    # label axis
    ax.set_xlabel("wavenumber", fontsize=fs)
    var_name, spectrum_unit = list(var_info.keys())[0], list(var_info.values())[0]
    ax.set_ylabel(f"Spectral power {var_name} [{spectrum_unit}]", fontsize=fs)
    ax.tick_params(axis="both", which="both", direction="out", labelsize=fs-2)
    ax.legend(fontsize=fs-2)
    

    if show:
        plt.show()

    if savefig:
        plt_fname = plt_fname + ".png" if not plt_fname.endswith(".png") else plt_fname
        print(f"Save plot in file '{plt_fname}'")
        plt.tight_layout()
        fig.savefig(plt_fname)
    plt.close(fig)


def create_model_compare_plots(ds_forecasts: list[xr.DataArray],ds_targets: list[xr.DataArray], ds_differences: list[xr.DataArray], labels: list[str], plt_fname: str, **kwargs):
    vars2plt = kwargs.pop("vars2plt", None)
    titles = kwargs.pop("titles", None)
    proj_data = kwargs.pop("proj_data", ccrs.RotatedPole(pole_longitude=-162.0, pole_latitude=39.25))
                      
    proj_plot = kwargs.pop("proj_plot", ccrs.PlateCarree())
    dims = kwargs.pop("dims", ["rlat", "rlon"])
    extent = kwargs.pop("extent", [3., 16.5, 43., 51.5])
    fs = kwargs.pop("fs", 16)
    figsize = kwargs.pop("figsize", (30, 6*len(labels)))
    # get levels and colorbars for 'normal' data plots and difference plots
    levels = kwargs.pop("levels", np.arange(-22., 42.1, 2))
    cbar_name = kwargs.pop("cbar_name", "jet")
    levels_diff = kwargs.pop("levels_diff", np.arange(-5.25, 5.01, 0.5))
    cbar_name_diff = kwargs.pop("cbar_name_diff", "PuOr_r")
    #seasons
    seasons = kwargs.pop("seasons","year")
    cbar_shrink = .8

    show = kwargs.pop("show",True)
    savefig = kwargs.pop("savefig",False)
    
    # auxiliary variables
    lvl, lvl_diff = np.asarray(levels), np.asarray(levels_diff)
    
    lat = ds_forecasts[0]['rlat'].values
    lon = ds_forecasts[0]['rlon'].values
    
    #lat, lon = var_now[dims[0]].values, var_now[dims[1]].values
    # construct array for edges of grid points
    dy, dx = np.round((lat[1] - lat[0]), 4), np.round((lon[1] - lon[0]), 4)
    lat_e, lon_e = np.arange(lat[0]-dy/2, lat[-1]+dy, dy), np.arange(lon[0]-dx/2, lon[-1]+dx, dx)

    # get colormap
    cmap, norm = get_cmap_norm(levels, cbar_name)
    cap_diff, norm_diff = get_cmap_norm(levels_diff, cbar_name_diff)
    
    # create plot objects
    fig, axs = plt.subplots(nrows=len(ds_forecasts), ncols=3, figsize=figsize, sharex=True, sharey=True,
                            subplot_kw={"projection": proj_plot})
    for idx,x in enumerate(ds_forecasts):
        forecast_data = ds_forecasts[idx]
        plot_fcst_data = axs[idx,0].pcolormesh(lon_e, lat_e, forecast_data, cmap=cmap, norm=norm, transform=proj_data,
                                        **kwargs)
        axs[idx,0].set_global()
        axs[idx,0].coastlines()
        axs[idx,0].add_feature(cfeature.BORDERS,linestyle=':')
        axs[idx,0].set_extent(extent, crs=proj_plot)
        axs[idx,0].set_title(labels[idx],fontsize=20)
        axs[idx,0].text(-0.04,0.4,'latitude [$^\circ$N]',va='bottom', ha='center',rotation='vertical',transform=axs[idx,0].transAxes)
        axs[idx,0].text(0.4,-0.07,'longitutde [$^\circ$E]',va='bottom', ha='center',rotation='horizontal',transform=axs[idx,0].transAxes)
        
        target_data = ds_targets[idx]
        plot_tar_data = axs[idx,1].pcolormesh(lon_e, lat_e, target_data, cmap=cmap, norm=norm, transform=proj_data,
                                        **kwargs)
        #axs[idx,1].set_title(titles[idx])
        axs[idx,1].set_global()
        axs[idx,1].coastlines()
        axs[idx,1].add_feature(cfeature.BORDERS,linestyle=':')
        axs[idx,1].set_extent(extent, crs=proj_plot)
        axs[idx,1].set_title("ground truth",fontsize=20)
        axs[idx,1].text(-0.04,0.4,'latitude [$^\circ$N]',va='bottom', ha='center',rotation='vertical',transform=axs[idx,1].transAxes)
        axs[idx,1].text(0.4,-0.07,'longitutde [$^\circ$E]',va='bottom', ha='center',rotation='horizontal',transform=axs[idx,1].transAxes)
        
        difference_data = ds_differences[idx]
        plot_diff = axs[idx,2].pcolormesh(lon_e, lat_e, difference_data, cmap=cap_diff, norm=norm_diff, transform=proj_data,
                                        **kwargs)
        axs[idx,2].set_global()
        axs[idx,2].coastlines()
        axs[idx,2].add_feature(cfeature.BORDERS,linestyle=':')
        axs[idx,2].set_extent(extent, crs=proj_plot)
        axs[idx,2].set_title(f"difference between {labels[idx]} and ground truth",fontsize=20)
        axs[idx,2].text(-0.04,0.4,'latitude [$^\circ$N]',va='bottom', ha='center',rotation='vertical',transform=axs[idx,2].transAxes)
        axs[idx,2].text(0.4,-0.07,'longitutde [$^\circ$E]',va='bottom', ha='center',rotation='horizontal',transform=axs[idx,2].transAxes)
    
    axs[len(labels)-1,1].text( 0.4,-0.3,
              f"All measurements are for the season : \"{seasons}\"" 
              if seasons != "year" 
              else "All measurements are for the year",
              va="bottom",
              ha="center",
              rotation="horizontal",
              transform=axs[len(labels)-1,1].transAxes,
                             fontsize=20)
             
    # add colorbars
    cbar = fig.colorbar(plot_fcst_data, ax=axs[0:len(labels),0:2], orientation="vertical", shrink=cbar_shrink, pad=.02, ticks=lvl[1::2], fraction=0.02)
    
    cbar.ax.tick_params(labelsize=fs-2)
        
    cbar = fig.colorbar(plot_diff, ax=axs[0:len(labels)], orientation="vertical", shrink=cbar_shrink, pad=.02, ticks=lvl[1::2], fraction=0.02)
    
    cbar.ax.tick_params(labelsize=fs-2)
   
    if show:
        plt.show()

    if savefig:
        plt_fname = plt_fname + ".png" if not plt_fname.endswith(".png") else plt_fname
        fig.savefig(plt_fname, bbox_inches="tight", dpi=300)
    plt.close(fig)
    
