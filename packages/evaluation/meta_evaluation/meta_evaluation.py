import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# auxiliary variable for logger
logger_module_name = f"main_meta_evaluation.{__name__}"
module_logger = logging.getLogger(logger_module_name)


def visualise_scorecard(
    scores: pd.DataFrame, ref_model: str, variable: str, savepath: str = None
):
    metrics = scores.score_name.unique()
    nmetrics = len(metrics)
    fig, axes = plt.subplots(
        nrows=1, ncols=nmetrics, figsize=(4 * nmetrics, 4), sharey=True
    )
    axiter = iter(axes)
    cbar_bounds = [-50, -20, -10, -5, -2, -1, 1, 2, 5, 10, 20, 50]
    col_order = ["YEAR", "DJF", "MAM", "JJA", "SON"]
    for metric, ax in zip(metrics, axiter):
        unit = metric2unit(metric=metric, variable=variable)
        pivot = scores.query(f"score_name == '{metric}'").pivot(
            columns="time", index="model", values="value"
        )
        pivot = pivot[col_order]
        # reference as first row
        pivot = pd.concat(
            [
                pivot.loc[[ref_model]],
                pivot.drop(ref_model),
            ]
        )
        labeldata = pivot.values
        heatmapdata = (
            (pivot - pivot.loc[ref_model]) / pivot.loc[ref_model] * 100
        )  # *100 to make it percentages

        im = heatmap(
            heatmapdata,
            pivot.index.values,
            pivot.columns.values,
            ax=ax,
            title=f"{metric.upper()} [{unit}]",
            cmap="coolwarm",
            norm=matplotlib.colors.BoundaryNorm(
                boundaries=cbar_bounds, ncolors=256, extend="both"
            ),
            alpha=0.85,
        )
        texts = annotate_heatmap(im, labels=labeldata, valfmt="{x:.2f}")

    cbar_ax = fig.add_axes([0.3, -0.01, 0.4, 0.04])
    cbar = fig.colorbar(
        im,
        cax=cbar_ax,
        orientation="horizontal",
        extend="both",
        ticks=cbar_bounds,
        label=f"Better <--    % difference vs reference model: {ref_model}    --> Worse",
    )
    plt.suptitle(variable.upper())
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, bbox_inches="tight")
    return None


def metric2unit(metric: str, variable: str):
    if variable == "t2m":
        return {
            "rmse": "K",
            "bias": "K",
            "grad_amplitude": "1",
            "me_std": "K",
            "ralsd": "1",
        }[metric]
    elif variable == "ws100m":
        return {
            "rmse": "m/s",
            "bias": "m/s",
            "grad_amplitude": "1",
            "me_std": "m/s",
            "ralsd": "1",
        }[metric]
    elif variable == "glob_rad":
        return {
            "rmse": "W/m²",
            "bias": "W/m²",
            "grad_amplitude": "1",
            "me_std": "W/m²",
            "ralsd": "1",
            "rmse_relative": "1",
            "bias_relative": "1",
            "fss_thres_50": "1",
            "fss_thres_100": "1",
            "fss_thres_300": "1",
            "fss_thres_500": "1",
        }[metric]


def heatmap(data, row_labels, col_labels, ax=None, title: str = None, **kwargs):
    if not ax:
        ax = plt.gca()
    if not title:
        title = ""
    im = ax.imshow(data, **kwargs)
    ax.set_xticks(
        range(data.shape[1]),
        labels=col_labels,
        rotation=0,
        ha="center",
        rotation_mode="anchor",
    )
    ax.set_yticks(range(data.shape[0]), labels=row_labels)
    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="lightgray", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(title)
    return im


def annotate_heatmap(im, labels=None, valfmt="{x:.2f}"):
    opts = dict(color="k", ha="center", va="center")

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            text = im.axes.text(j, i, valfmt(labels[i, j], None), **opts)
            texts.append(text)
    return texts


def load_aggregate_scores(config: dict, basedir: str) -> pd.DataFrame:
    scores = []
    for _, modelpath in config["models"].items():
        scores_file = Path(
            basedir,
            modelpath,
            "metric_files",
            "aggregate_scores",
            "scores.csv",
        )
        scores_iter = pd.read_csv(scores_file, index_col=0)
        scores.append(scores_iter)
    return pd.concat(scores)


def load_metric_time_series(
    config: dict, basedir: str, metric: str, fname: str
) -> xr.DataArray:
    metric_list = []
    for modelname, modelpath in config["models"].items():
        metric_file = Path(
            basedir,
            modelpath,
            "metric_files",
            "temporal_evaluation",
            fname,
        )
        metric_iter = xr.open_dataset(metric_file)[f"{metric}_mean"]
        metric_iter = metric_iter.assign_coords(model=modelname)
        metric_list.append(metric_iter)
    return xr.concat(metric_list, dim="model")


def plot_multimodel_metric_line(
    data: xr.DataArray,
    metric: dict,
    time_period: str,
    plt_fname: str,
    varname: str = "T2m",
    x_coord: str = "hour",
    value_range: tuple = (0.0, 3.0),
    **kwargs,
):
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
                    - title: title of the plot
                    - "linestyle": linestyle of plot, default: "k-"
                    - "error_color": color of error bounds, default: "blue"
                    - "value_range": range of y-axis, default: (0., 4.)
                    - "fs": font size of labels, default: 16
                    - "ref_line": reference line to be plotted, default: None
                    - "ref_linestyle": linestyle of reference line, default: "k--"
                    - other valid arguments of ax.plot
    """

    # get some plot parameters
    title = time_period
    fs = kwargs.pop("fs", 16)
    ref_line = kwargs.pop("ref_line", None)
    ref_linestyle = kwargs.pop("ref_linestyle", "k--")

    fig, (ax) = plt.subplots(1, 1)

    linestyles = ["C0-d", "C1-x", "C2-o", "C3-d", "C4-x", "C5-o"]

    # plot data
    for i, model in enumerate(data.model.values):
        model_data = data.sel(model=model)
        metric_name, metric_unit = list(metric.keys())[0], list(metric.values())[0]
        ax.plot(
            model_data[x_coord].values,
            model_data.values,
            linestyles[i],
            lw=0.8,
            markersize=5,
            **kwargs,
        )
    # make legend
    ax.legend(data.model.values, loc="upper right")

    if ref_line is not None:
        nval = np.shape(data[x_coord].values)[0]
        ax.plot(data[x_coord].values, np.full(nval, ref_line), ref_linestyle)
    ax.set_ylim(*value_range)
    ax.set_xlim(0, 23)

    # label axis
    ax.set_xlabel("daytime [UTC]", fontsize=fs)
    ax.set_ylabel(f"{metric_name} {varname} [{metric_unit}]", fontsize=fs)
    ax.tick_params(axis="both", which="both", direction="out", labelsize=fs - 2)

    ax.set_title(title.upper(), size=fs)

    # enable grid
    ax.grid(alpha=0.5)

    # save plot and close figure
    plt_fname = plt_fname + ".png" if not plt_fname.endswith(".png") else plt_fname
    fig.savefig(plt_fname, bbox_inches="tight")
    plt.tight_layout()
    fig.savefig(plt_fname)
    plt.close(fig)
    return None
