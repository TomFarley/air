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

from fire.interfaces.interfaces import read_csv

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def calc_heatflux(t, temperatures, s_analsis_path, tile_properties):
    """"""
    import theodor
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
    # For hints as to meanings to theodor arguments see:
    # https://users.euro-fusion.org/openwiki/index.php/THEODOR#theo_mul_33.28.29
    qtile, extra_results = theodor.theo_mul_33(temperatures, t, s_analsis_path, d_target, alpha_bot, alpha_top, diff, lam,
                                               aniso, x_Tb=x_Tb, y_Tb=y_Tb, test=True, verbose=verbose)

if __name__ == '__main__':
    pass