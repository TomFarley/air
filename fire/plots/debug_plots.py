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

from fire.plots.figures import figure_spatial_res_max, figure_spatial_res_x, figure_spatial_res_y, plot_spatial_res_hist
from fire.image_processing import find_outlier_pixels
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def debug_spatial(data, aspect='equal'):
    fig, axes = plt.subplots(2, 3, num='raycast', figsize=(13, 13))
    axes = axes.flatten()
    # Frame data
    ax = axes[0]
    n = data['frame_data'].n.median()
    frame_data = data['frame_data'].sel(n=n)
    hot_pixels, frame_data = find_outlier_pixels(frame_data, tol=3)
    ax.imshow(frame_data, interpolation='none', cmap='gray', aspect=aspect)
    ax.plot(hot_pixels[1], hot_pixels[0], ls='', marker='o', color='r', ms=1)
    # data['frame_data'].sel(n=n).plot(ax=ax)
    # R coord
    ax = axes[1]
    # key = 'R_im'
    # key = 'phi_deg_im'
    # key = 'x_im'
    key = 'y_im'
    cbar_label = {'R_im': r'$R$ coordinate [m]',
                  'phi_deg_im': r'$\phi$ coordinate [$^\circ$]',
                  'x_im': r'$x$ coordinate [m]',
                  'y_im': r'$y$ coordinate [m]'
                  }
    data[key].plot.imshow(ax=ax, add_colorbar=True, origin='upper',
                             cbar_kwargs=dict(label=cbar_label[key], extend='both'))
    ax.set_aspect(aspect)
    # Spatial res hist
    ax = axes[2]
    plot_spatial_res_hist(data, ax=ax, show=False, save_fn=None)

    # Spatial res
    ax = axes[3]
    figure_spatial_res_x(data, ax=ax, show=False, save_fn=None, aspect=aspect)
    ax = axes[4]
    figure_spatial_res_y(data, ax=ax, show=False, save_fn=None, aspect=aspect)
    ax = axes[5]
    figure_spatial_res_max(data, ax=ax, show=False, save_fn=None, aspect=aspect)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    pass