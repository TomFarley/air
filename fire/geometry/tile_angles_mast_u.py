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
from scipy import interpolate
import matplotlib.pyplot as plt

from fire.interfaces.basic_io import read_csv
from fire.plotting.plot_tools import create_poloidal_cross_section_figure, get_fig_ax
from fire.geometry.geometry import cartesian_to_toroidal

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

def read_tile_surface_coords():

    path = Path('../input_files/mast_u/tile_surface_coords/')
    files = ['T1_surface_MU01303.csv', 'T2_surface_MU00859.csv', 'T3_surface_MU00860.csv', 'T4_surface_MU00861.csv',
             'T5A_surface_MU00746.csv', 'T5B_surface_MU00747.csv', 'T5C_surface_MU00748.csv', 'T5D_surface_MU00749.csv']
    tile_data = {}
    headers = ['x', 'y', 'z']
    for file in files:
        fn = path/file
        data = read_csv(fn, names=headers)
        r, phi, theta = cartesian_to_toroidal(data['x'], data['y'], data['z'], angles_in_deg=True)
        data['R'] = r
        data['phi'] = phi
        data['theta'] = theta

        tile_id = file.split('_')[0]
        tile_data[tile_id] = data

    return tile_data

def plot_tile_surfaces_rz(tile_data, tile_ids=None, ax=None, show=False):
    if ax is None:
        fig, ax = create_poloidal_cross_section_figure()

    if tile_ids is None:
        tile_ids = list(tile_data.keys())

    for tile_id in tile_ids:
        data = tile_data[tile_id]  # x, y, z

        ax.plot(data['R'], data['z'], ls='', marker='o', markersize=2, alpha=0.8, label=tile_id)

    if show:
        plt.legend()
        plt.show()

def plot_tile_surface_3d(tile_Data, tile_ids=None, ax=None, show=False):
    if ax is None:
        fig, ax, ax_passed = get_fig_ax(ax, num='Tile surfaces', dimensions=3)

    if tile_ids is None:
        tile_ids = list(tile_data.keys())

    for tile_id in tile_ids:
        data = tile_data[tile_id]  # x, y, z
        x, y, z = data['x'], data['y'], data['z']

        # ax.plot(data['x'], data['y'], data['z'], ls='', marker='o', markersize=2, label=tile_id)
        grid_x, grid_y = np.mgrid[np.min(x):np.max(x):500j, np.min(y):np.max(y):500j]
        grid_z = interpolate.griddata(np.array([x, y]).T, z, (grid_x, grid_y), fill_value=np.nan)
        surf = ax.plot_surface(grid_x, grid_y, grid_z, label=tile_id)
        # Deal with mpl bug - tmp
        surf._facecolors2d = surf._facecolors3d
        surf._edgecolors2d = surf._edgecolors3d

    if show:
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    tile_data = read_tile_surface_coords()
    plot_tile_surface_3d(tile_data, tile_ids=None, show=True)
    plot_tile_surfaces_rz(tile_data, tile_ids=None, show=True)
    pass