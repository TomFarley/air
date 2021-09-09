#!/usr/bin/env python

"""


Created: 
"""

import logging
from typing import Union, Iterable, Sequence, Tuple, Optional, Any, Dict
from pathlib import Path
from copy import copy
from cycler import cycler

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from fire.plotting import plot_tools
from matplotlib import pyplot as plt

from fire.misc.utils import make_iterable

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

def plot_movie_intensity_stats(data_movie, ax=None, bit_depth=14, meta_data=None, removed_frames=None, num=None,
                               show=True, save_fn=None):

    num = f'Movie average intensity' if num is None else num
    fig, ax, ax_passed = plot_tools.get_fig_ax(ax, num=num)

    data_movie = copy(data_movie)

    # if 't' not in data_movie.dims:
    #     data_movie = data_movie.swap_dims({data_movie.dims[0]: 't'})
    try:
        t = np.array(data_movie['t'])
        n = np.array(data_movie['n'])
        ax.set_xlabel(r'$t$ [s]')
    except Exception as e:
        t = np.arange(len(data_movie))
        n = np.arange(len(data_movie))

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
    plot_tools.annotate_axis(ax, r'$t_{int}=$'+f'{meta_data.get("exposure")*1e3:0.3g} ms', loc='bottom right')
    plot_tools.legend(ax)

    ax_n = plot_tools.add_second_x_scale(ax, x_axis_values=n,
                                         label='$n_{frame}$', y_values=data_max)

    if removed_frames:
        # Label ends of movie that are discarded due discontinuous intensities etc
        ax_n.axvline(x=removed_frames['start'], ls=':', color='k', label='clipped bad frames from start')
        ax_n.axvline(x=len(data_movie)-1-removed_frames['end'], ls=':', color='k', label='clipped bad frames from end')

    plot_tools.show_if(show=show, close_all=False, tight_layout=True)
    plot_tools.save_fig(save_fn, fig=fig, save=(save_fn is not None))

    return fig, ax

def plot_passed_temporal_stats(stat_profiles, stat_labels, stats=None, t=None, ax=None,
                               line_styles=(':', '-.', '--', '-.', ':', (0, (2, 5)), (0, (2, 9)))):
    if t is None:
        data = stat_profiles[list(stat_labels.values())[0]]
        if isinstance(data, xr.DataArray):
            t = np.array(data['t'])
        else:
            t = np.arange(len(data))
            # raise ValueError('t value not supplied')

    fig, ax, ax_passed = plot_tools.get_fig_ax(ax, num=f'Temporal stats')

    line_styles = (cycler(ls=line_styles))
    # line_styles = (cycler(ls=['-', '--', '-.', ':', (0, (2, 5)), (0, (2, 9))]))
    for stat, ls in zip(stats, line_styles):
        ax.plot(t, stat_profiles[stat_labels[stat]], label=stat, alpha=0.7, **ls)


def plot_temporal_stats(data_2d, t=None, ax=None, t_axis=0,
                        stats=('max', '99.0%', 'mean', '1.0%', 'min'),
                        ls=(':', '--', '-', '--', ':', (0, (2, 5)), (0, (2, 9))),
                        bit_depth=None, meta_data=None, times_of_interest=None, y_label=None, num=None,
                        show=True, save_fn=None, roll_width=None, roll_reduce_func='mean', roll_center=True):
    from fire.physics.physics_parameters import calc_1d_profile_rolling_stats, calc_2d_profile_param_stats
    from fire.misc.data_structures import swap_xarray_dim

    stats = make_iterable(stats)
    num = 'Temporal stats' if num is None else num
    fig, ax, ax_passed = plot_tools.get_fig_ax(ax, num=num, figsize=(12, 8))

    data_2d = copy(data_2d)

    stat_profiles, stat_labels = calc_2d_profile_param_stats(data_2d, stats=stats, coords_reduce=('y_pix', 'x_pix'),
                                                             roll_width=roll_width,
                                                             roll_reduce_func=roll_reduce_func, roll_center=roll_center)
    # calc_1d_profile_rolling_stats()


    plot_passed_temporal_stats(stat_profiles, stat_labels, stats=stats, t=t, ax=ax, line_styles=make_iterable(ls))

    n = np.arange(data_2d.shape[t_axis])

    try:
        data_2d = swap_xarray_dim(data_2d, 't')
        t = data_2d['t']
        n = data_2d['n']
    except Exception as e:
        pass

    if t is not None:
        ax.set_xlabel(r'$t$ [s]')
    else:
        t = n
        ax.set_xlabel(r'$x$ [arb]')

    if bit_depth is not None:
        ax.axhline(2**bit_depth, label='max DL', color='k', ls='--')

    ax.set_ylabel(y_label)

    plot_tools.annotate_providence(ax, meta_data=meta_data, annotate=(meta_data is not None))
    # plot_tools.annotate_axis(ax, r'$t_{int}=$'+f'{meta_data.get("exposure")*1e3:0.3g} ms', loc='bottom right')
    plot_tools.legend(ax)

    ax_n = plot_tools.add_second_x_scale(ax, x_axis_values=n, label='$n_{frame}$')

    plot_tools.save_fig(save_fn, fig=fig, save=(save_fn is not None))
    plot_tools.show_if(show=show, close_all=False, tight_layout=True)

    return fig, ax, ax_n

if __name__ == '__main__':
    pass