# SPDX-FileCopyrightText: 2024 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC)
#
# SPDX-License-Identifier: MIT

"""
Methods for creating plots.
"""

__author__ = "Ankit Patnala"
__email__ = "a.patnala@fz-juelich.de"
__date__ = "2024-05-16"
__update__ = "2024-05-16"
import xarray as xr
import numpy as np

import os

from plotting import plot_metric_line,plot_skills,plot_comparison_maps,plot_power_spectra

class Config:
    def __init__(self,base_folder,variable,models,seasons,metric,**kwargs):
        if variable not in ['t2m','wind','solar_irradiance']:
            raise ValueError(f"variable should be one of ['t2m','wind','solar_irradiance'] but {variable} was provided")

        if seasons not in ["year","DJF","JJA","MAM","SON"]:
            raise ValueError("seasons should be one of [year, DJF, JJA, MAM, SON] but {seasons} was provided")
        
        if metric not in ["bias","grad_amplitude","me_std","rmse"]:
            raise ValueError("metric should be one of [bias, grad_amplitude, me_std, rmse] but {metric} was provided")
        self.base_folder = base_folder
        self.variable = variable
        self.models = models
        self.seasons = seasons
        self.metric = metric

    def __repr__(self):
        return str(self.__dict__)

def convert_date_time_to_underscore_format(month_date_hour_dict):
    month = month_date_hour_dict.pop("month",None)
    date = month_date_hour_dict.pop("date",None)
    hour = month_date_hour_dict.pop("hour",None)
    strf_format = ""
    text = ""
    if month is not None:
        strf_format += "-%b"
        text += f"-{month}"
    if date is not None:
        strf_format += "-%d"
        text += f"-{date}"
    if hour is not None:
        strf_format += "-%H:%M"
        text += f"-{hour}:00"

    return strf_format,text

def score_line_plots(config : Config, uncertainty = False,**kwargs):
    nc_files_mean  = []
    for model in config.models:
        model_path = os.path.join(config.base_folder,f"{model}_benchmark_{config.variable}","metric_files")
        nc_files_mean.append(os.path.join(model_path,f"eval_{config.metric}_{config.seasons}.nc"))
    mean_array = []
    array_down = [] if uncertainty else None
    array_up = [] if uncertainty else None
    for nc_file_mean in nc_files_mean:
        data_array = xr.open_dataset(nc_file_mean)
        mean_array.append(data_array[f"{config.metric}_mean"])
        if uncertainty:
            array_down.append(
                    data_array[f"{config.metric}_mean_boot"].quantile(
                        0.01,
                        dim='iboot'))
            array_up.append(
                data_array[f"{config.metric}_mean_boot"].quantile(
                    0.99,
                    dim='iboot'))

    labels = kwargs.pop("labels",config.models)
    metric_dict = kwargs.pop("metric_dict",None)
    plt_fname = kwargs.pop("plt_fname", "metric_line")
    varname = config.variable
    plot_metric_line(mean_array, array_up, array_down, labels,metric_dict, plt_fname,varname,show=True,**kwargs)

def skill_box_plot(config,ref_model,**kwargs):

    assert(ref_model in config.models)

    nc_files_mean = []
    ref_model_path = os.path.join(config.base_folder,f"{ref_model}_benchmark_{config.variable}","metric_files")
    ref_data = xr.open_dataset(os.path.join(ref_model_path,f"eval_{config.metric}_{config.seasons}.nc"))
    modified_labels = []
    for model in config.models:
        if model != ref_model:
            model_path = os.path.join(config.base_folder,f"{model}_benchmark_{config.variable}","metric_files")
            nc_files_mean.append(os.path.join(model_path,f"eval_{config.metric}_{config.seasons}.nc"))
            modified_labels.append(model)
    mean_array_boot = []
    for nc_file_mean in nc_files_mean:
        data_array = xr.open_dataset(nc_file_mean)
        # To-Do: 
        # Set A_perf correctly rather than assuming zero as perfect score
        mean_array_boot.append(np.expand_dims(np.mean(1 - data_array[f"{config.metric}_mean_boot"]/ref_data[f"{config.metric}_mean_boot"]  ,axis=0),axis=0))
    data = np.concatenate(mean_array_boot)
    plt_fname = kwargs.pop("plt_fname","box_plot")
    plot_skills(data.T,plt_fname,labels=modified_labels,metric= config.metric,**kwargs)


def model_comparison_plot(config):
    file_paths = []
    variable = config.variable
    for model in config.models:
        file_path = os.path.join(config.base_folder,f"{model}_benchmark_{variable}",f"downscaled_{variable}_{model}.nc")
        file_paths.append(file_path)
    
    data_array = []

    if 'datetime' in config.__dict__.keys():
        strf_format,text = convert_date_time_to_underscore_format(config.datetime)
    else:
        config.datetime = None

    for file_path in file_paths:
        dataset = xr.open_dataset(file_path)
        ground_truth = dataset[f"{variable}_ref"]-273.15
        forecast = dataset[f"{variable}_fcst"]-273.15
        if config.datetime is None:
            if config.seasons != "year":
                ground_truth = ground_truth[ground_truth["time.season"] == config.seasons]
                forecast = forecast[forecast["time.season"] == config.seasons]
        else:
            ground_truth = ground_truth[ground_truth["time"].dt.strftime(strf_format) == text]
            forecast = forecast[forecast["time"].dt.strftime(strf_format) == text]

        dataset = xr.merge([ground_truth.mean(dim="time"),forecast.mean(dim="time")])
        data_array.append(dataset)

    plot_comparison_maps(data_array,"compare_plot",savefig=False,show=True,models=config.models)


def spectra_plot(config,var_info,**kwargs):
    nc_files_spectral = []
    for model in config.models:
        model_path = os.path.join(config.base_folder,f"{model}_benchmark_{config.variable}","spectral_analysis")
        nc_files_spectral.append(os.path.join(model_path,f"{config.variable}_power_spectrum_{config.seasons}.nc"))

    datasets_ps = []
    for i,dataset in enumerate(nc_files_spectral):
        ds = xr.open_dataset(dataset)
        if i==0:
            ds_ref = xr.Dataset(data_vars={"reference":ds[f"{config.variable}_ref"]},
                    coords={'wavenumber':ds['wavenumber']})
            datasets_ps.append(ds_ref)
            del ds_ref
        ds = xr.Dataset(data_vars={f"{config.models[i]}":ds[f"{config.variable}_fcst"]},
                coords={'wavenumber':ds['wavenumber']})
        datasets_ps.append(ds)

    datasets_ps = xr.merge(datasets_ps)

    plot_power_spectra(datasets_ps,var_info,['reference']+config.models,"spectral_energy",**kwargs)





