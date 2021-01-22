#!/usr/bin/env python

"""
Primary analysis workflow for MAST-U/JET IR scheduler analysis code.

Created: 10-10-2019
"""

import logging
from typing import Union, Iterable, Tuple, Optional
from copy import copy
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# TODO: Move meta data defaults to json file
meta_defaults_default = {
    't': {'label': 'Time', 'units': 's', 'symbol': '$t$',
          'description': 'Camera frame time from start of discharge'},
    'S': {'label': 'Divertor tile S coordinate', 'units': 'm', 'symbol': '$S$',
          'description': 'Spatial coordinate along surface of PFCs'},
    'R': {'label': 'Major radius', 'units': 'm', 'symbol': '$R$',
          'description': 'Radial distance from centre of machine'},
    'T': {'label': 'Divertor tile temperature', 'units': '$^\circ$C', 'symbol': '$T$',
          'description': 'Divertor tile temperature surface temperature measured by IR camera'},
    'q': {'label': 'Divertor tile incident heat flux', 'units': 'MWm$^{-2}$', 'symbol': '$q_\perp$',
          'description': 'Divertor tile incident heat flux measured by IR camera'},
    'frame_data': {'label': 'Digital level', 'units': 'counts', 'symbol': '$DL$',
              'description': 'Camera sensor pixel digital level signal, dependent on photon flux'},
    'x': {'label': 'x', 'units': 'm', 'symbol': '$x$',
                  'description': 'Machine cartesian X coordinate'},
    'y': {'label': 'y', 'units': 'm', 'symbol': '$y$',
                      'description': 'Machine cartesian Y coordinate'},
    'z': {'label': 'z', 'units': 'm', 'symbol': '$z$',
                          'description': 'Machine cartesian Z coordinate'},
    'n': {'label': 'Frame number', 'units': 'count', 'symbol': '$n_{frame}$',
                  'description': 'Camera frame number index (integer from 0)'},
    'x_pix': {'label': 'x pixel coordinate', 'units': 'count', 'symbol': '$x_{pix}$',
              'description': 'Camera sensor x (left to right) pixel coordinate (integer)'},
    'y_pix': {'label': 'y pixel coordinate', 'units': 'count', 'symbol': '$y_{pix}$',
              'description': 'Camera sensor y (top to bottom) pixel coordinate (integer)'},
    'phi_deg': {'label': 'Toroidal coordinate', 'units': '$^\circ$', 'symbol': '$\phi$',
                  'description': 'Machine toroidal coordinate in degrees (phi=0 along x axis)'},
    's_global': {'label': 'Divertor tile S coordinate', 'units': 'm', 'symbol': '$S$',
                  'description': 'Spatial coordinate along surface of PFCs'},
    'surface_id': {'label': 'Surface id', 'units': 'count', 'symbol': '$ID_{surface}$',
                  'description': 'Integer ID specifying what machine component/tile etc is visible at that pixel'},

}
# TODO: Add dict of alternative names for each variable in meta_defaults_default eg 'S', 's_global'

for key in meta_defaults_default:
    meta_defaults_default[key]['long_name'] = meta_defaults_default[key]['symbol']  # Set longname for xarray plots

def init_data_structures() -> Tuple[dict, dict, xr.Dataset, dict]:
    """Return empty data structures for population by and use in fire code

    :return: settings, files, data, meta_data
    :rtype: dict, dict, xr.Dataset, dict
    """

    settings = {}
    files = {}
    data = xr.Dataset()
    meta_data = {}

    return settings, files, data, meta_data

def attach_standard_meta_attrs(data, varname='all', replace=False, key=None):

    if varname == 'all':
        for var in data.data_vars:
            data = attach_standard_meta_attrs(data, varname=var)

    if key is None:
        key = varname

    if (key in meta_defaults_default):
        new_attrs = copy(meta_defaults_default[key])

        if isinstance(data, xr.Dataset):
            data_array = data[varname]
        else:
            data_array = data
            varname = data_array.name

        if not replace:
            # Keep existing meta data values
            new_attrs.update(data_array.attrs)

        data_array.attrs.update(new_attrs)
        logger.debug(f'Added standard meta data for "{varname}" (replace={replace})')
    else:
        logger.debug(f'No standard meta data for "{varname}" ("{key}"). '
                       f'Defaults for: {[k for k in meta_defaults_default.keys()]}')

    return data