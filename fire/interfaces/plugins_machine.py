#!/usr/bin/env python

"""


Created: 
"""

import logging
from typing import Union, Iterable, Sequence, Tuple, Optional, Any, Dict, Callable
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

coord = Union[float, np.ndarray]

def get_machine_location_labels(x_im: coord, y_im: coord, z_im: coord,
                                machine_plugins: Dict[str, Callable], plugin_subset: Optional[Sequence]=None, **kwargs):
    """Return a dict of labels corresponding to the supplied coordinates (e.g. sector number etc.)

    Args:
        x_im            : Array (potentially 2D image) of cartesian spatial x coordinates (m)
        y_im            : Array (potentially 2D image) of cartesian spatial y coordinates (m)
        z_im            : Array (potentially 2D image) of cartesian spatial z coordinates (m)
        machine_plugins : Dict of plugin functions to be called
        plugin_subset   : List of names/keys of plugin functions to call
                          (if None tries calling 3 defaults: 'sector', 's_coord_global', 's_coord_path')
        **kwargs        : Optional additional keywords to pass to plugin functions

    Returns: Dict containing  for supplied coordinates (if plugin fucs
             supplied)

    """
    if plugin_subset is None:
        if 'location_labels' in machine_plugins:
            plugin_subset = machine_plugins['location_labels']
        else:
            plugin_subset = ['sector', 's_coord_global', 's_coord_path']
    data = {}
    for plugin in plugin_subset:
        if plugin in machine_plugins:
            try:
                func = machine_plugins[plugin]
                data[plugin] = func(x_im, y_im, z_im, **kwargs)
            except KeyError as e:
                pass
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