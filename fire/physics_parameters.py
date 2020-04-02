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

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

"""Old IDL sched iranalysis.pro input parameters
  ;variables
  ;sh = shot
  ; trange= time range, array e.g [0.0,0.1]
  ;ldef - returned from get_ldef, which uses loc - the ldef path definitions file looked up with loc
  ;loc - analysis path name e.g. 'louvre4'
  ;t - returned times
  ;s - returned radius
  ;h - returned temperature?
  ;qpro - returned heat flux
  ;numsatpix - total number of saturated pixels on analysis path
  ;alphaconst - alpha constant, can be array, defined in rit2air_comb_view.pro
  ;tsmooth -
  ;tbgnd -
  ;targetrate -
  ;nline -
  ;aug - run for ASDEX, defunct?
  ;print - set flag for output to PS
"""

# TODO: Move module defaults to json files
params_dict_default = {'required':
                           ['alpha_const'],
                       'optional':
                            []
                       }

legacy_values = {23586:
                     {"AIR_ALPHACONST_ISP": 70000.0,
                     "AIR_ALPHACONST_ISP_ELM": 30000.0,
                     "AIR_CAMERA VIEW_ISP": 2,
                     "AIR_CAMERA VIEW_OSP": 4,
                     "AIR_ERRORFLAG": 110

                      }
                 }

# Sentinel for default keyword arguments
module_defaults = object()

def check_input_params_complete(data, params_dict=module_defaults):
    if params_dict is None:
        params_dict = params_dict_default
    for param in params_dict['required']:
        if param not in data:
            raise KeyError(f'Analysis input parameter "{param}" missing from:\n{data}')

def attach_meta_data(data, meta_data_dict=None):
    raise NotImplementedError

def calc_horizontal_path_anulus_areas(r_path):
    """Return areas of horizontal annuli around machine at each point along the analysis path.

    Calculates the toroidal sum of horizontal surface area of tile around the machine between each radial point
    around the machine.
    Post processing corrections are required to account for the fact that the tiles surfaces are not actually
    horizontal due to both tilt in toroidal plane and toroidal tilt of tile (i.e. tile surface hight is func of
    toroidal andle z(phi))

    In MAST the horizontal divertor simplified the final area calculation to
        2 * pi * R * dR;

    Args:
        r_path: Radial coordinates of each point along the analysis path

    Returns: Areas of horizontal annuli around machine at each point along the analysis path.

    """
    dr = r_path[2:] - r_path[0:-2]
    np.insert(dr, )
    das = 2.0 * np.pi * (s + ldef(0, 0)) * das2;
    # why not 1 / 2 as one Rib group only?

def calc_physics_params(data):


if __name__ == '__main__':
    pass

