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

from fire import fire_paths
from fire.geometry import get_s_coord_global_r, get_s_coord_path_ds
from fire.interfaces.plugins import get_plugins

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def get_machine_location_labels(x_im, y_im, z_im, machine_plugins=None, **kwargs):
    data = {}
    for plugin in ['sector']:
        if plugin in machine_plugins:
            func = machine_plugins[plugin]
            data[plugin] = func(x_im, y_im, z_im, **kwargs)
    return data

def get_s_coord_global(x_im, y_im, z_im, machine_plugins=None, **kwargs):
    """Return spatial "s" tile coodinate for each pixel in the image (some pixels may be nan).

    Use machine specific "s_coord_global" function if available, else default to returning major radius (only ~valid
    for flat divertors).

    Args:
        x_im            : x spatial coord for each pixel
        y_im            : y spatial coord for each pixel
        z_im            : z spatial coord for each pixel
        machine_plugins : Dict of functions from which the fucntion keyed "s_coord_global" is used if available
        **kwargs        : Aditional argumemnts to s_coord_global function

    Returns: Tile "s" coordinate for each pixel in image

    """
    if (machine_plugins is not None) and ('s_coord_global' in machine_plugins):
        func = machine_plugins['s_coord_global']
    else:
        func = get_s_coord_global_r  # default fallback
    s_global = func(x_im, y_im, z_im, **kwargs)
    return s_global

def get_s_coord_path(x_path, y_path, z_path, machine_plugins, **kwargs):
    if (machine_plugins is not None) and ('s_coord_global' in machine_plugins):
        func = machine_plugins['s_coord_path']
    else:
        func = get_s_coord_path_ds  # default fallback
    s_path = func(x_path, y_path, z_path, **kwargs)
    return s_path

if __name__ == '__main__':
    pass