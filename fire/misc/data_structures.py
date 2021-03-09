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

from fire.misc.utils import make_iterable

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
    'i_path': {'label': 'Array index along analysis path', 'units': '', 'symbol': '$i_{path}$',
                  'description': 'Array index along analysis path (alternative to R/s_global)'},
    'surface_id': {'label': 'Surface id', 'units': 'count', 'symbol': '$ID_{surface}$',
                  'description': 'Integer ID specifying what machine component/tile etc is visible at that pixel'},
    'temperature_peak': {'label': 'Peak temperature', 'units': '$^\circ$C', 'symbol': '$T_{peak}$',
                  'description': 'Peak target temperature along analysis path as a function of time'},
    'heat_flux_peak': {'label': 'Peak heat flux', 'units': 'MWm$^{-2}$', 'symbol': '$q_{\perp,peak}$',
                  'description': 'Peak target heat flux along analysis path as a function of time'},
    'heat_flux_r_peak': {'label': 'Target location', 'units': 'm', 'symbol': '$R_{target}$',
                  'description': 'Radial location of peak target heatflux'},
    'temperature_r_peak': {'label': 'Peak temperature location', 'units': 'm', 'symbol': '$R_{T,max}$',
                      'description': 'Radial location of peak target temperature'},
    'xpx/clock/lwir-1': {'label': 'LWIR 1 trigger signal', 'units': 'V', 'symbol': '$V_{LWIR1, trig}$',  # TODO: move to uda_utils
                         'description': 'Voltage of IRCAM1 LWIR HL04 datac loop trigger signal (xpx/clock/lwir-1)'},

}
# TODO: Add dict of alternative names for each variable in meta_defaults_default eg 'S', 's_global'
param_aliases = {
    'temperature': 'T',
    's_global': 'S',
    's_local': 'S',
    'frame_data_nuc': 'frame_data'
}

for key in meta_defaults_default:
    meta_defaults_default[key]['long_name'] = meta_defaults_default[key]['symbol']  # Set longname for xarray plots

def init_data_structures() -> Tuple[dict, dict, xr.Dataset, xr.Dataset, dict, dict]:
    """Return empty data structures for population by and use in fire code

    :return: settings, files, data, meta_data
    :rtype: dict, dict, xr.Dataset, dict
    """

    settings = {}
    files = {}
    data_image = xr.Dataset()
    data_path = xr.Dataset()
    meta_data = {}
    meta_runtime = {}  # Used to collect together objects eg calcam.Calibration that may be hard to serialise

    return settings, files, data_image, data_path, meta_data, meta_runtime


def movie_data_to_dataarray(frame_data, frame_times=None, frame_nos=None, meta_data=None, name='frame_data'):
    """Return frame data in xarray.DataArray object

    Args:
        frame_data  : Array of camera digit level data with dimensions [t, y, x]
        frame_times : Array of frame times
        frame_nos   : Array of frame numbers

    Returns: DataArray of movie data

    """
    if frame_nos is None:
        frame_nos = np.arange(frame_data.shape[0])
    if frame_times is None:
        frame_times = copy(frame_nos)
    if meta_data is None:
        meta_data = {}

    frame_data = xr.DataArray(frame_data, dims=['t', 'y_pix', 'x_pix'],
                              coords={'t': frame_times, 'n': ('t', frame_nos),
                                      'y_pix': np.arange(frame_data.shape[1]),
                                      'x_pix': np.arange(frame_data.shape[2])},
                              name=name)
    # Default to indexing by frame number
    frame_data = frame_data.swap_dims({'t': 'n'})
    if 'frame_data' in meta_data:
        frame_data.attrs.update(meta_data['frame_data'])
        frame_data.attrs['label'] = frame_data.attrs['description']
    else:
        logger.warning(f'No meta data supplied for coordinate: {"frame_data"}')

    coords = ['n', 't', 'x_pix', 'y_pix']
    for coord in coords:
        if coord in meta_data:
            frame_data[coord].attrs.update(meta_data[coord])
            # UDA requires 'label' while xarray uses description
            frame_data[coord].attrs['label'] = frame_data[coord].attrs['description']
        else:
            logger.warning(f'No meta data supplied for coordinate: {coord}')

    return frame_data

def attach_standard_meta_attrs(data, varname='all', replace=False, key=None):

    if varname == 'all':
        for var in data.data_vars:
            data = attach_standard_meta_attrs(data, varname=var)

    if key is None:
        key = varname

    if key in param_aliases:
        key = param_aliases[key]

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


def to_image_dataset(data, key='data'):
    if isinstance(data, xr.Dataset):
        dataset = data
    elif isinstance(data, xr.DataArray):
        dataset = xr.Dataset({data.name: data})
    elif isinstance(data, np.ndarray):
        # Use calcam convention: image data is indexed [y, x], but image shape description is (nx, ny)
        ny, nx = data.shape
        x_pix = np.arange(nx)
        y_pix = np.arange(ny)
        dataset = xr.Dataset(coords={'x_pix': x_pix, 'y_pix': y_pix})
        # data = xr.Dataset({'data': (('y_pix', 'x_pix'), data)}, coords={'x_pix': x_pix, 'y_pix': y_pix})
        dataset[key] = (('y_pix', 'x_pix'), data)
        dataset['x_pix'].attrs.update({
            'long_name': '$x_{pix}$',
            'units': '',
            'description': 'Camera x pixel coordinate'})
        dataset['y_pix'].attrs.update({
            'long_name': '$y_{pix}$',
            'units': '',
            'description': 'Camera y pixel coordinate'})
        dataset = attach_standard_meta_attrs(dataset, varname=key)
        # TODO: Move to utils/data_structures?
        # TODO: fix latex display of axis labels
        # TODO: use this func in calcam_calibs get_surface_coords
    else:
        raise ValueError(f'Unexpected image data type {data}')
    return dataset


def get_reduce_coord_axis_keep(data, coord_reduce):
    # Get preserved dimension used for xr.apply_ufunc
    ndim = data.values.ndim
    if ndim == 2:
        axis_reduce = list(data.dims).index(coord_reduce)
        coord_keep = list(data.dims)
        coord_keep.pop(data.dims.index(coord_reduce))
        coord_keep = coord_keep[0]
        axis_keep = list(data.dims).index(coord_keep)
    elif ndim == 1:
        coord_keep = None
        axis_keep = None
        axis_reduce = None
    else:
        raise ValueError(f'Too many dimensions: {data}')

    return coord_keep, axis_keep, axis_reduce


def reduce_2d_data_array(data_array, func_reduce, coord_reduce, func_args=()):
    """Apply a numpy ufunc to reduce 2D DataArray dimensionality to 1D for plotting etc

    Args:
        data_array: Input 2D dataarray
        func_reduce: np ufunc, eg np.mean
        coord_reduce: Coordinate name to apply ufunc across (ie average across time, 't')
        func_args: Any additional args to pass to ufunc eg percentile to pass to np.percentile

    Returns: DataArray with coord_reduce compressed by func_reduce

    """
    func_args = make_iterable(func_args)
    coord_keep, axis_keep, axis_reduce = get_reduce_coord_axis_keep(data_array, coord_reduce)
    input_core_dims = ((coord_reduce,), ) + tuple(() for i in np.arange(len(func_args)))

    data_array_reduced = xr.apply_ufunc(func_reduce, data_array, *func_args,
                                                input_core_dims=input_core_dims, kwargs=dict(axis=axis_keep))
    return data_array_reduced