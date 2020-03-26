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
import matplotlib.pyplot as plt

from fire.interfaces.interfaces import read_csv

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def calc_heatflux(t, temperatures, analysis_path, material_properties, visible_materials):
    """"""
    from fire import theodor
    t = np.array(t)

    # TODO: Check time axis is uniform
    dt = np.diff(t)
    if not np.all(np.isclose(dt, dt[0], rtol=1e-3)):
        raise ValueError(f'Time axis of data supplied to THEODOR is not uniform. dt: {set(dt)}')
    # TODO: Check analysis path for jumps/tile gaps etc

    # TODO generalise to other path names (i.e. other than 'path')
    material_ids = set(np.array(analysis_path['material_id_path']))
    if len(material_ids) > 1:
        raise NotImplementedError
    # TODO: Loop over sections of path with different material properties or tile gaps etc
    xpix_path, ypix_path = analysis_path['x_pix_path'], analysis_path['y_pix_path']
    temperature_path = np.array(temperatures[:, ypix_path, xpix_path])
    s_path = np.array(analysis_path['s_path'])  # spatial coordinate along tile surface
    material_id = list(material_ids)[0]
    material_name = visible_materials[material_id]
    theo_kwargs = material_properties[material_name]

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
    if False:
        # tmp imports for theodor debugging
        import faulthandler, gc
        faulthandler.enable()
        gc.disable()

    heat_flux, extra_results = theodor.theo_mul_33(temperature_path, t, s_path, test=True, verbose=True,
                                                   **theo_kwargs)
    #               d_target, alpha_bot, alpha_top, diff, lam, aniso, x_Tb=x_Tb, y_Tb=y_Tb,

    return heat_flux, extra_results

if __name__ == '__main__':
    pass