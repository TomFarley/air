# -*- coding: future_fstrings -*-
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
from scipy import stats
import matplotlib.pyplot as plt

from fire.interfaces.interfaces import read_csv
from fire.misc.utils import safe_len

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


def calc_heatflux(t, temperatures, path_data, path_name, material_properties, visible_materials):
    """"""
    from fire import theodor
    t = np.array(t)
    path = path_name

    # Check theodor time axis is uniform
    dt = np.diff(t)
    dt_mode = stats.mode(dt).mode
    mask_const_dt = np.isclose(dt, dt_mode, atol=9e-7, rtol=5e-4)
    if not np.all(mask_const_dt):
        raise ValueError(f'Time axis of data supplied to THEODOR is not uniform. mode={dt_mode}. dt other: '
                         f'{dt[~mask_const_dt]}')
    # TODO: Check analysis path for jumps/tile gaps etc

    # TODO generalise to other path names (i.e. other than 'path')
    material_ids = np.array(path_data[f'material_id_{path}'])
    material_ids = set(material_ids[~np.isnan(material_ids)])
    if len(material_ids) == 0:
        raise ValueError(f'No surface materials identified along analysis path: {path}')
    elif len(material_ids) > 1:
        raise NotImplementedError
    # TODO: Loop over sections of path with different material properties or tile gaps etc
    xpix_path, ypix_path = path_data[f'x_pix_{path}'], path_data[f'y_pix_{path}']
    temperature_path = np.array(temperatures[:, ypix_path, xpix_path])
    s_path = np.array(path_data[f's_global_{path}'])  # spatial coordinate along tile surface
    material_id = list(material_ids)[0]
    material_name = visible_materials[material_id]
    theo_kwargs = material_properties[material_name]

    # TODO: Understand when two element values should be used
    if safe_len(theo_kwargs['alpha_top_org']) == 2:
        theo_kwargs['alpha_top_org'] = theo_kwargs['alpha_top_org'][0]

    if temperature_path.shape[0] != len(s_path):
        raise ValueError(f'Spatial dimention of temperature path data ({temperature_path.shape}) does not match s_path '
                         f'dimension ({len(s_path)}). Mismatch will cause theodor to use integer index s values!')

    # alpha_bot = alphas['tile_bottom']
    # alpha_top = alphas['tile_surface']
    #
    # d_target=0.040
    # diff = np.array([ 70.63, 48.25, 37.78])*1.0e-6
    # lam = np.array([174.9274, 133.1148, 110.4595])
    # #diff = np.array([240.87, 61.53, 34.86])*1e-6    # mm^2/s    Valerias JET data, 01/12/2008
    # #lam  = np.array([305.28,175.68,117.12])         # W/m/K     Valerias JET data
    # aniso=1.00
    # alpha_bot=200.0
    # acl=alphavector if alphavector is not None else np.array([1.])
    # alpha_top = alpha*acl if alpha is not None else 220000.0*acl


    # For hints as to meanings to theodor arguments see:
    # https://users.euro-fusion.org/openwiki/index.php/THEODOR#theo_mul_33.28.29
    # NOTE: If location[x] is not the same size as the x dimension of data[x,t], the code uses the array indices as location[x].
    if False:
        # tmp imports for theodor debugging
        import faulthandler, gc
        faulthandler.enable()
        gc.disable()

    """
    Required arguments:
    data            - float temperature array (location,time)
    time            - temporal coordinates of data[1]
    location        - spatial coordinates of data[0]
    d_target        - target thickness in m
    alpha_bot       - heat transmission coefficient at the bottom of the tile
    alpha_top_org   - heat transmission coefficient at the top of the tile. Not sure when this should have two elements 
    diff            - heat diffusion coefficient at [0,500,1000 C] m^2/s
    lam             - heat conduction coefficient (at [0,500,1000 C]) W/m/K
    aniso           - anisotropy of vertical and tangential heat conduction. Ratio of 1.0 -> isotropic
    
    NOTES:
    - The remaining arguments are for visualisation/debugging and can be ignored
    - If location[x] is not the same size as the x dimension of data[x,t], the code uses the array indices as location[x].
    """

    heat_flux, extra_results = theodor.theo_mul_33(temperature_path, t, s_path, test=True, verbose=True,
                                                   **theo_kwargs)
    #               d_target, alpha_bot, alpha_top, diff, lam, aniso, x_Tb=x_Tb, y_Tb=y_Tb,

    return heat_flux, extra_results

if __name__ == '__main__':
    pass