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

import pyuda
import fire
from fire.scripts.review_analysed_shot import read_data_for_pulses_pickle
from fire.plotting import plot_tools, debug_plots

logger = logging.getLogger(__name__)
logger.propagate = False

def check_time_axis_rescale():
    pulse = 43583
    data = read_data_for_pulses_pickle('rit', pulse)
    path_data = data[pulse][0]['path_data']

    # fig = plt.figure()
    # plt.contourf(path_data['heat_flux_path0'])

    plt.figure()
    plt.plot(path_data['heat_flux_peak_path0'].t,path_data['heat_flux_peak_path0'].values, label='raw', ls=':')

    if pulse >= 43543 and pulse <= 43621:
        t_scale_factor = 0.616342 / 0.56744
        plt.plot(path_data['heat_flux_peak_path0'].t*t_scale_factor,path_data['heat_flux_peak_path0'].values,
                 label='scaled', alpha=0.7)


    #get the dalpha data
    client=pyuda.Client()
    da=client.get('/xim/da/hm10/t', 43583)

    plt.plot(da.time.data, da.data*5, label=r'$D_{\alpha}$', alpha=0.7)
    plt.legend()
    plt.show()

def compare_t2_t5_heat_flux():
    camera = 'rit'
    signal_ir = 'heat_flux_path0'
    # pulse = 43583
    pulse = 43610
    # pulse = 43620
    # pulse = 43624

    machine = 'mast_u'
    meta = dict(camera=camera, pulse=pulse, machine=machine, signal=signal_ir)

    align_peaks = False
    # align_peaks = True
    # t_window = 10e-3
    t_window_t2 = 0.0
    t_window_t5 = 0.0

    t_window_t2 = 0.005
    # t_window_t5 = 0.004

    # t_tile2 = 0.12559
    t_tile2 = 0.16
    # t_tile2 = 0.115

    # t_tile5 = 0.3526
    t_tile5 = 0.315
    # t_tile5 = 0.25

    r_t2_bounds = [0.540, 0.905]
    r_t5_bounds = [1.395, 1.745]

    data = read_data_for_pulses_pickle(camera, pulse, machine)
    path_data = data[pulse][0]['path_data']
    path_data = path_data.swap_dims({'i_path0': 'R_path0'})

    fig, axes, ax_passed = plot_tools.get_fig_ax(ax_grid_dims=(1, 1), sharex=True, axes_flatten=True)
    ax = axes

    r_coord = 'R_path0'
    r = path_data[r_coord]
    t = path_data['t']

    if pulse >= 43543 and pulse <= 43621:
        t_scale_factor = 0.616342 / 0.56744
    else:
        t_scale_factor = 1
    # t = t * t_scale_factor
    # path_data['t'] = t


    # r_t2[0] = r.values.min()
    # r_t5[1] = r.values.max()

    heat_flux = path_data[signal_ir]

    if not t_window_t2:
        profile_t2 = heat_flux.sel(t=t_tile2, method='nearest')
    else:
        t2_mask = (t >= t_tile2-t_window_t2/2) & (t <= t_tile2+t_window_t2/2)
        profile_t2 = heat_flux.sel(t=t2_mask, method='nearest').mean(dim='t')
    t2_mask = (r >= r_t2_bounds[0]) & (r <= r_t2_bounds[1])
    profile_t2 = profile_t2.sel(R_path0=t2_mask)
    r_t2 = profile_t2[r_coord]

    if not t_window_t5:
        profile_t5 = heat_flux.sel(t=t_tile5, method='nearest')
    else:
        t5_mask = (t >= t_tile5-t_window_t5/2) & (t <= t_tile5+t_window_t5/2)
        profile_t5 = heat_flux.sel(t=t5_mask, method='nearest').mean(dim='t')
    t5_mask = (r >= r_t5_bounds[0]) & (r <= r_t5_bounds[1])
    profile_t5 = profile_t5.sel(R_path0=t5_mask)
    r_t5 = profile_t5[r_coord]

    if align_peaks:
        r_t2 = r_t2 - r_t2[profile_t2.argmax(dim=r_coord)]
        r_t5 = r_t5 - r_t5[profile_t5.argmax()]

    ax.plot(r_t2, profile_t2, label=rf'Tile 2 ($t={t_tile2:0.3f}$ s)')
    ax.plot(r_t5, profile_t5, label=rf'Tile 5 ($t={t_tile5:0.3f}$ s)')

    ax.set_xlabel(f'$R$ [m]')
    ax.set_ylabel(f'{heat_flux.symbol} [{heat_flux.units}]')
    ax.title.set_visible(False)

    if True:
        ax.set_yscale('log')
        ax.set_ylim([profile_t5.min(), None])

    plot_tools.legend(ax, loc='center right', box=False)

    plot_tools.annotate_providence(ax, meta=meta, box=False)

    if align_peaks:
        fn = f'{camera}_{pulse}_T2T5_aligned.png'
    else:
        fn = f'{camera}_{pulse}_T2T5_vs_R.png'

    path_fn = fire.fire_paths['root'] / 'figures' / fn
    plot_tools.save_fig(path_fn, verbose=True)

    plot_tools.show_if(True, tight_layout=True)

    # 2d heaflux map
    fig, ax, ax_passed = plot_tools.get_fig_ax(ax_grid_dims=(1, 1), sharex=True, axes_flatten=True)

    debug_plots.debug_plot_profile_2d(data_paths=path_data, param='heat_flux', ax=ax, robust=True,
                                          machine_plugins='mast_u', show=False)
    if t_window_t2:
        ax.axhline(y=t_tile2-t_window_t2, ls=':', color='tab:blue', lw=2)
        ax.axhline(y=t_tile2+t_window_t2, ls=':', color='tab:blue', lw=2)

        ax.axhline(y=t_tile5 - t_window_t5, ls='--', color='tab:orange', lw=2, alpha=0.8)
        ax.axhline(y=t_tile5 + t_window_t5, ls='--', color='tab:orange', lw=2, alpha=0.8)
    else:
        ax.axhline(y=t_tile2, ls='--', color='tab:blue', lw=2)
        ax.axhline(y=t_tile5, ls='--', color='tab:orange', lw=2)


    plot_tools.show_if(True, tight_layout=True)


if __name__ == '__main__':
    import pyuda
    client = pyuda.Client()

    compare_t2_t5_heat_flux()

    check_time_axis_rescale()
    pass