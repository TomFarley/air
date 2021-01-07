#!/usr/bin/env python

"""


Created: 
"""

import logging
from typing import Union, Iterable, Sequence, Tuple, Optional, Any, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from fire.plotting.plot_tools import get_fig_ax

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


def figure_path_1d(data, key, slice=None, ax=None, plot_kwargs=None):
    """Plot profile of variable given by key along analysis path

    Args:
        data: Dataset containing plot data
        key: Name of variable in dataset to plot
        slice: (Temporal) slice to plot
        ax: Matplotlib axis to plot to
        plot_kwargs: Additional formatting args to pass to plt.plot

    Returns: ax

    """
    fig, ax, ax_passed = get_fig_ax(ax)

    if slice is None:
        slice = {'n': np.floor(np.median(data['n']))}
    kws = {'color': None}
    if isinstance(plot_kwargs, dict):
        kws.update(plot_kwargs)

    data_plot = data[key]
    if data_plot.ndim > 1:
        if 'n' not in data_plot.dims:
            data_plot = data_plot.swap_dims({'t': 'n'})
        data_plot = data_plot.sel(slice)

    data_plot.plot.line(ax=ax, **kws)
    ax.title.set_fontsize(10)

    return ax

def figure_path_2d(data, key, slice=None, ax=None, plot_kwargs=None):
    fig, ax, ax_passed = get_fig_ax(ax)

    if slice is None:
        raise NotImplementedError('Subset of frames/time range')
        # slice = {'n': np.floor(np.median(data['n']))}

    kws = {'cmap': None}
    if isinstance(plot_kwargs, dict):
        kws.update(plot_kwargs)

    data_plot = data[key]
    if slice is not None:
        data_plot = data_plot.sel(slice)

    data_plot.plot(ax=ax, robust=True, **kws)
    ax.title.set_fontsize(10)

    return ax

if __name__ == '__main__':
    pass