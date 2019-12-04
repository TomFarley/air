#!/usr/bin/env python

"""


Created: 
"""

import logging
from typing import Union, Iterable, Tuple, Optional
from pathlib import Path

import numpy as np
import theodor

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def calc_heatflux(t, temperatures, s_analsis_path, alphas):
    """"""
    d_target
    alpha_bot = alphas['tile_bottom']
    alpha_top = alphas['tile_surface']

    # TODO: Check time axis is uniform

    qtile, extra_results = theodor.theo_mul_33(temperatures, t, s_analsis_path, d_target, alpha_bot, alpha_top, diff, lam,
                                               aniso, x_Tb=x_Tb, y_Tb=y_Tb, test=True, verbose=verbose)

if __name__ == '__main__':
    pass