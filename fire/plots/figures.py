# -*- coding: future_fstrings -*-
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
from matplotlib import ticker, colors

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

dpi = 90

def figure_spatial_res(data, key='spatial_res_max', cbar_label=None, ax=None,
                       cmap='gray_r', log_cmap=True, axes_off=False, origin='upper', aspect='equal',
                       save_fn=None, show=False):
    if ax is None:
        fig, ax = plt.subplots(1, 1, num=key)
        ax_passed = False
    else:
        ax_passed = True

    scale_factor = 1e3
    data_plot = (data[key] * scale_factor)
    if log_cmap:
        norm = colors.LogNorm()
        data_plot = data_plot.where(data_plot > 0, np.nan)  # Remove zeros to avoid divide by zero
    else:
        norm = None

    data_plot.plot.imshow(ax=ax,
                          cmap=cmap, norm=norm, interpolation='none', origin=origin,
                          cbar_kwargs=dict(label=cbar_label, extend='both'))

    if axes_off:
        ax.set_axis_off()
    if aspect:
        ax.set_aspect(aspect)
    if not ax_passed:
        plt.tight_layout()

    if save_fn:
        plt.savefig(save_fn, bbox_inches='tight', transparent=True, dpi=dpi)
        logger.info(f'Saved spatial resolution figure to: {save_fn}')
    if show:
        plt.show()

def figure_spatial_res_max(data, ax=None, log_cmap=True, aspect='equal', axes_off=True, save_fn=None, show=False):
    key = 'spatial_res_max'
    cbar_label = 'Spatial resolution (max) [mm]'
    figure_spatial_res(data, key, cbar_label=cbar_label, ax=ax, log_cmap=log_cmap, axes_off=axes_off, aspect=aspect,
                       save_fn=save_fn, show=show)

def figure_spatial_res_x(data, ax=None, log_cmap=True, aspect='equal', axes_off=True, save_fn=None, show=False):
    key = 'spatial_res_x'
    cbar_label = 'Spatial resolution (x) [mm]'
    figure_spatial_res(data, key, cbar_label=cbar_label, ax=ax, log_cmap=log_cmap, axes_off=axes_off, aspect=aspect,
                       save_fn=save_fn, show=show)

def figure_spatial_res_y(data, ax=None, log_cmap=True, aspect='equal', axes_off=True, save_fn=None, show=False):
    key = 'spatial_res_y'
    cbar_label = 'Spatial resolution (y) [mm]'
    figure_spatial_res(data, key, cbar_label=cbar_label, ax=ax, log_cmap=log_cmap, axes_off=axes_off, aspect=aspect,
                       save_fn=save_fn, show=show)

def plot_spatial_res_hist(data, ax=None, log_y=True, save_fn=None, show=False):
    if ax is None:
        fig, ax = plt.subplots(1, 1, num='spatial_res max hist')
        ax_passed = False
    else:
        ax_passed = True
    ax.hist(data['spatial_res_max'].values.flatten(), bins=200)
    ax.set_xlabel('spatial res (max) [m]')
    if log_y:
        ax.set_yscale('log')
    if not ax_passed:
        plt.tight_layout()
        ax.set_aspect('equal')
    if save_fn:
        plt.savefig(save_fn, bbox_inches='tight', transparent=True, dpi=dpi)
        logger.info(f'Saved y spatial resolution figure to: {save_fn}')
    if show:
        plt.show()
if __name__ == '__main__':
    pass