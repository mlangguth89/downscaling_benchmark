# SPDX-FileCopyrightText: 2025 Earth System Data Exploration (ESDE), Jülich Supercomputing Center (JSC); Gesosphere Austria (GSA)
#
# SPDX-License-Identifier: MIT

"""
Contains all methods and classes used in main_evaluation.py.
"""

__author__ = "Sebastian Lehner, Michael Langguth"
__email__ = "sebastian.lehner@geosphere.at, m.langguth@fz-juelich.de"
__date__ = "2025-10-03"
__update__ = "2025-10-03"

import os
import logging
import matplotlib.pyplot as plt
import numpy as np

logger_module_name = f"main_inference.{__name__}"
module_name = os.path.basename(__file__).rstrip(".py")

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
