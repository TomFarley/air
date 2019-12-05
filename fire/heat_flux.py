#!/usr/bin/env python

"""


Created: 
"""

import logging
from typing import Union, Iterable, Tuple, Optional, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import theodor

from fire.interfaces.interfaces import read_csv

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def get_tile_properties(machine: str, pulse: Union[int, str], analysis_path: xr.Dataset,
                        input_paths: Sequence[Union[Path, str]]) -> Tuple[xr.DataArray, pd.DataFrame]:
    """Return tile properties at each point along analysis path

    Args:
        machine         : Machine/tokamak under analysis
        pulse           : Pulse number used to identify time of tile state
        analysis_path   : Spatial coordinates of path along which to return tile properties
        input_paths     : Paths to look for tile properties files

    Returns: ([Array of tile names for each coord point], [DataFrame of tile properties])

    """
    # TODO: Look up tile for each point along analysis path using (R, z) box lookup from file

    analysis_path_tiles = lookup_tiles(analysis_path['R'], analysis_path['z'])
    analysis_path_tiles = xr.DataArray(analysis_path_tiles,
                                       coords={'R_path': analysis_path['R'], 'z_path': analysis_path['x']})

    tile_properties = []
    for tile in set(analysis_path_tiles):
        tile_data = read_csv()
        tile_properties.append(tile_data)
    tile_properties = xr.DataArray(coords=analysis_path.coords)
    # TODO: Segment path according to changes in tile properties
    # tile_properties['i_path']

    return analysis_path_tiles, tile_properties

def lookup_tiles(r, z, path_fn_tile_coords, raise_on_no_tile_info=True):
    # TODO: Read path_fn_tile_coords
    tile_names = np.full_like(r, '', dtype=object)
    tile_coords = read_csv(path_fn=path_fn_tile_coords, index_col='tile_name')



    no_tile_info_mask = tile_name == ''
    if any(no_tile_info_mask):
        tile_names[no_tile_info_mask] = np.nan
        if raise_on_no_tile_info:
            raise ValueError(f'Analysis path contains {np.sum(no_tile_info_mask)}/{len(r)} points without tile info:\n'
                             f'r={r[no_tile_info_mask]}\nz={z[no_tile_info_mask]}')
    return tile_names

def calc_heatflux(t, temperatures, s_analsis_path, tile_properties):
    """"""
    d_target
    alpha_bot = alphas['tile_bottom']
    alpha_top = alphas['tile_surface']

    d_target=0.040
    diff = np.array([ 70.63, 48.25, 37.78])*1.0e-6
    lam = np.array([174.9274, 133.1148, 110.4595])
    #diff = np.array([240.87, 61.53, 34.86])*1e-6    # mm^2/s    Valerias JET data, 01/12/2008
    #lam  = np.array([305.28,175.68,117.12])         # W/m/K     Valerias JET data
    aniso=1.00
    alpha_bot=200.0
    acl=alphavector if alphavector is not None else np.array([1.])
    alpha_top = alpha*acl if alpha is not None else 220000.0*acl

    # TODO: Check time axis is uniform

    qtile, extra_results = theodor.theo_mul_33(temperatures, t, s_analsis_path, d_target, alpha_bot, alpha_top, diff, lam,
                                               aniso, x_Tb=x_Tb, y_Tb=y_Tb, test=True, verbose=verbose)

if __name__ == '__main__':
    pass