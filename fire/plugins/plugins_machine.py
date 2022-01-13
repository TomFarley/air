#!/usr/bin/env python

"""


Created: 
"""

import logging
from typing import Union, Sequence, Optional, Dict, Callable
from pathlib import Path

import numpy as np

from fire.geometry.s_coordinate import get_s_coord_global_r, get_s_coord_path_ds
from fire.interfaces import user_config

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

coord = Union[float, np.ndarray]

def get_machine_coordinate_labels(x_im: coord, y_im: coord, z_im: coord,
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
            plugin_subset = machine_plugins['location_labels_im']
        else:
            plugin_subset = ['sector', 's_global', 's_coord_path']
    data = {}
    for plugin in plugin_subset:
        if plugin in machine_plugins:
            try:
                func = machine_plugins[plugin]
                data[plugin] = func(x_im, y_im, z_im, **kwargs)
            except KeyError as e:
                pass
    # TODO: Add s_global_no_nans variables with interpolated out nans?
    return data

def get_s_coord_global(x_im, y_im, z_im, machine_plugins=None, **kwargs):
    """Return spatial "s" tile coodinate for each pixel in the image (some pixels may be nan).

    Use machine specific "s_coord_global" function if available, else default to returning major radius (only ~valid
    for flat divertors).
    This 's' coordinate is considered 'global' as it is predefined for all (R, Z) surfaces as apposed to a 'local' s
    starting at 0m along a specific path.

    Args:
        x_im            : x spatial coord for each pixel
        y_im            : y spatial coord for each pixel
        z_im            : z spatial coord for each pixel
        machine_plugins : Dict of functions from which the fucntion keyed "s_coord_global" is used if available
        **kwargs        : Additional arguments to s_coord_global function

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

def get_machine_plugins(machine='mast_u'):
    import fire
    from fire.interfaces import interfaces

    from fire.plugins import plugins
    config, config_groups, path_fn = user_config.get_user_fire_config()
    base_paths = config_groups['fire_paths']

    # Load machine plugins
    machine_plugin_paths = config['paths_input']['plugins']['machine']
    machine_plugin_attrs = config['plugins']['machine']['module_attributes']
    machine_plugins, machine_plugins_info = plugins.get_compatible_plugins(machine_plugin_paths,
                                                attributes_required=machine_plugin_attrs['required'],
                                                attributes_optional=machine_plugin_attrs['optional'],
                                                plugins_required=machine, plugin_type='machine',
                                                                           base_paths=base_paths)
    machine_plugins, machine_plugins_info = machine_plugins[machine], machine_plugins_info[machine]
    fire.active_machine_plugin = (machine_plugins, machine_plugins_info)

    return machine_plugins


if __name__ == '__main__':
    pass