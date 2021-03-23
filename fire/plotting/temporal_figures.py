#!/usr/bin/env python

"""


Created: 
"""

import logging
from typing import Union, Iterable, Sequence, Tuple, Optional, Any, Dict
from pathlib import Path
from copy import copy

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from fire.plotting import plot_tools
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

def plot_temporal_profile(profile_data, param, path_name=None, ax=None, mask=None, show=True, save_fn=None,
                          label=None, **plot_kwargs):

    fig, ax, ax_passed = plot_tools.get_fig_ax(ax, num=f'{param} temporal profile')

    key = param.format(path=path_name) if (path_name is not None) else param
    data = profile_data[key]

    if mask is not None:
        data[~mask] = np.nan

    if label is True:
        label = data.attrs.get('label', data.attrs.get('symbol', None))

    kwargs = dict(ls='-', alpha=0.8, label=label)
    kwargs.update(plot_kwargs)

    data.plot(ax=ax, **kwargs)  # , label=data.attrs.get('symbol', None)

    plot_tools.show_if(show=show, close_all=False)
    plot_tools.save_fig(save_fn, fig=fig, save=(save_fn is not None))

    return fig, ax

def plot_movie_intensity_stats(data_movie, ax=None, bit_depth=14, meta_data=None, show=True, save_fn=None):

    fig, ax, ax_passed = plot_tools.get_fig_ax(ax, num=f'Movie average intensity')

    data_movie = copy(data_movie)

    # if 't' not in data_movie.dims:
    #     data_movie = data_movie.swap_dims({data_movie.dims[0]: 't'})
    try:
        t = data_movie['t']
        ax.set_xlabel(r'$t$ [s]')
    except Exception as e:
        t = np.arange(len(data_movie))

    data_av = data_movie.mean(axis=(1, 2))
    data_min = data_movie.min(axis=(1, 2))
    data_max = data_movie.max(axis=(1, 2))
    data_1p = np.percentile(data_movie, 1, axis=(1, 2))
    data_99p = np.percentile(data_movie, 99, axis=(1, 2))

    ax.axhline(2**bit_depth, label='max DL', color='k', ls='--')

    ax.plot(t, data_max, label='max', ls=':')
    ax.plot(t, data_99p, label='99%', ls='--')
    ax.plot(t, data_av, label='mean', ls='-')
    ax.plot(t, data_1p, label='1%', ls='--')
    ax.plot(t, data_min, label='min', ls=':')

    ax.set_ylabel(r'Raw frame intensity [DL]')

    plot_tools.annotate_providence(ax, meta_data=meta_data, annotate=(meta_data is not None))
    plot_tools.legend(ax)

    plot_tools.add_second_x_scale(ax, x_axis_values=data_movie.coords['n'], label='$n_{frame}$', y_values=data_max)

    plot_tools.show_if(show=show, close_all=False, tight_layout=True)
    plot_tools.save_fig(save_fn, fig=fig, save=(save_fn is not None))

    return fig, ax

if __name__ == '__main__':
    pass