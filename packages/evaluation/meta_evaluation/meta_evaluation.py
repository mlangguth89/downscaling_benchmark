import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def visualise_scorecard(scores: pd.DataFrame, variable: str, savepath: str = None):
    metrics = scores.score_name.unique()
    nmetrics = len(metrics)
    fig, axes = plt.subplots(nrows=1, ncols=nmetrics, figsize=(4*nmetrics,4), sharey=True);
    axiter = iter(axes)
    cbar_bounds = [-50, -20, -10, -5, -2, -1, 1, 2, 5, 10, 20, 50]
    col_order = ["YEAR", "DJF", "MAM", "JJA", "SON"]
    for metric, ax in zip(metrics, axiter):
        unit = metric2unit(metric=metric, variable=variable)
        pivot = scores.query(f"score_name == '{metric}'").pivot(columns="time", index="model", values="value")
        pivot = pivot[col_order]
        labeldata = pivot.values
        heatmapdata = (pivot - pivot.loc[config["ref_model"]])/pivot.loc[config["ref_model"]]*100 # *100 to make it percentages
    
        im = heatmap(
            heatmapdata,
            pivot.index.values,
            pivot.columns.values,
            ax=ax,
            title=f"{metric.upper()} [{unit}]",
            cmap="coolwarm",
            norm=matplotlib.colors.BoundaryNorm(boundaries=cbar_bounds, ncolors=256, extend="both"),
            alpha=0.85,
        )
        texts = annotate_heatmap(im, data=labeldata, valfmt="{x:.2f}", threshold=10)
    
    cbar_ax = fig.add_axes([0.3, 0.02, 0.4, 0.04])
    cbar = fig.colorbar(
        im,
        cax=cbar_ax,
        orientation="horizontal",
        extend="both",
        ticks=cbar_bounds,
        label=f"Better <--    % difference vs reference model: {config['ref_model']}    --> Worse",
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
    im = ax.imshow(data, **kwargs);
    ax.set_xticks(range(data.shape[1]), labels=col_labels,
                  rotation=0, ha="center", rotation_mode="anchor")
    ax.set_yticks(range(data.shape[0]), labels=row_labels)
    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="lightgray", linestyle='-', linewidth=3)
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
            text = im.axes.text(j, i, valfmt(labels[i, j], None), **opts);
            texts.append(text)
    return texts

def load_aggregate_scores(config: dict):
    scores = []
    for model in config["models"]:
        scores_file = Path(config["base_folder"], config["variable"], model, "metric_files", "aggregate_scores", "scores.csv")
        scores_iter = pd.read_csv(scores_file, index_col=0)
        scores.append(scores_iter)
    return pd.concat(scores)
