#!/usr/bin/env python

"""
Primary analysis workflow for MAST-U/JET IR scheduler analysis code.

Created: 10-10-2019
"""

import logging
import re
from typing import Union, Iterable, Tuple, Optional
from copy import copy
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import stats

from fire.misc import utils
from fire.misc.utils import make_iterable

logger = logging.getLogger(__name__)

# TODO: Move meta data defaults to json file
meta_defaults_default = {
    't': {'label': 'Time', 'units': 's', 'symbol': '$t$', 'class': 'time',
          'description': 'Camera frame time from start of discharge'},
    'S': {'label': 'Divertor tile S coordinate', 'units': 'm', 'symbol': '$S$',
          'description': 'Spatial coordinate along surface of PFCs'},
    'R': {'label': 'Major radius', 'units': 'm', 'symbol': '$R$',
          'description': 'Radial distance from centre of machine'},
    'T': {'label': 'Divertor tile temperature', 'units': '$^\circ$C', 'symbol': '$T$',
          'description': 'Divertor tile temperature surface temperature measured by IR camera'},
    'q': {'label': 'Divertor tile incident perpendicular heat flux', 'units': 'MWm$^{-2}$', 'symbol': '$q_\perp$',
          'description': 'Divertor tile incident perpendicular heat flux (q_\perp) measured by IR camera'},
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
    'phi': {'label': 'Toroidal coordinate', 'units': 'radians', 'symbol': '$\phi$',
                  'description': 'Machine toroidal coordinate in radians (phi=0 along x axis)'},
    'phi_deg': {'label': 'Toroidal coordinate', 'units': '$^\circ$', 'symbol': '$\phi$',
                  'description': 'Machine toroidal coordinate in degrees (phi=0 along x axis)'},
    'theta': {'label': 'Poloidal angle coordinate', 'units': '$radians$', 'symbol': '$\theta',
                  'description': 'Machine poloidal coordinate in radians (theta=0 in radial direction)'},
    's_global': {'label': 'Divertor tile S coordinate', 'units': 'm', 'symbol': '$s$',
                  'description': 'Spatial coordinate along surface of PFCs'},
    's_local': {'label': 'Local divertor tile S coordinate for analysis path', 'units': 'm', 'symbol': '$s_{local}$',
                  'description': 'Local divertor tile S coordinate for analysis path. This is zero at the start of '
                                 'the analysis path and increases with distance along the analysis path'},
    'i_path': {'label': 'Array index along analysis path', 'units': 'count', 'units_plot': '',
               'symbol': '$i_{path}$',
               'description': 'Integer array index along analysis path (alternative to R/s_global)'},
    'surface_id': {'label': 'Surface id', 'units': 'count', 'symbol': '$ID_{surface}$',
                  'description': 'Integer ID specifying what machine component/tile etc is visible at that pixel'},
    'temperature_peak': {'label': 'Peak temperature', 'units': '$^\circ$C', 'symbol': '$T_{peak}$',
                  'description': 'Peak target temperature along analysis path as a function of time'},
    'heat_flux_amplitude_peak_global': {'label': 'Peak heat flux', 'units': 'MWm$^{-2}$', 'symbol': '$q_{\perp,peak}$',
                  'description': 'Peak target heat flux along analysis path as a function of time'},
    'heat_flux_R_peak': {'label': 'Radius of target peak heat flux', 'units': 'm', 'symbol': '$R_{q,max}$',
                  'description': 'Radial location of peak target heat flux'},
    'heat_flux_s_global_peak': {'label': 's coordinate of target peak heat flux', 'units': 'm', 'symbol': '$s_{q,max}$',
                         'description': 'Divertor surface "s" coordinate location of peak target heat flux'},
    'heat_flux_total': {'label': 'Heat flux integrated over divertor surface area', 'units': 'MW',
                        'symbol': '$q_{total}$',
                        'description': 'Heat flux integrated over divertor surface area (for this divertor). Wetted '
                                       'area correction applied.'},
    'heat_flux_n_peaks': {'label': 'Number of local peaks in heat flux for each time point', 'units': 'count',
                        'symbol': '$n_{q, peaks}$',
                        'description': 'Number of local peaks in heat flux for each time point'},
    'temperature_R_peak': {'label': 'Radius of target peak temperature', 'units': 'm', 'symbol': '$R_{T,max}$',
                      'description': 'Radial location of peak target temperature'},
    'xpx/clock/lwir-1': {'label': 'LWIR 1 trigger signal', 'units': 'V', 'symbol': '$V_{LWIR1, trig}$',  # TODO: move to uda_utils
                         'description': 'Voltage of IRCAM1 LWIR HL04 datac loop trigger signal (xpx/clock/lwir-1)'},
    '/xpx/clock/LWIR-1': {'label': 'LWIR 1 trigger signal', 'units': 'V', 'symbol': '$V_{LWIR1, trig}$',
                         'description': 'Voltage of IRCAM1 LWIR HL04 datac loop trigger signal (xpx/clock/lwir-1)',},
                        }
# TODO: Add dict of alternative names for each variable in meta_defaults_default eg 'S', 's_global'
param_aliases = {
    'temperature': 'T',
    'heat_flux': 'q',
    's_global': 'S',
    's_local': 'S',
    'frame_data_nuc': 'frame_data',  # NUC corrected movie data
    'nuc_frame': 'frame_data',  # NUC frame
    'i': 'i_path',  # When '_path\d+' has been subbed out
    'i_path0': 'i_path',
    'i_path1': 'i_path',
    'i_path2': 'i_path'  # TODO: Generalise to other path numbers etc
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

def attach_standard_meta_attrs(data, varname='all', replace=False, key=None, substitutions=None):
    """

    Args:
        data: Dataset or DataArray. If DataArray, varname is not required
        varname: Name of variable in dataset to attach meta data to
        replace: Whether to replace/overwrite existing meta data
        key: Optional key used to look up standard meta data. This may be a generalised version of varname

    Returns: Updated data with data.attrs updated with standard meta data

    """

    if varname == 'all':
        for var in data.data_vars:
            data = attach_standard_meta_attrs(data, varname=var)

    if key is None:
        key = varname

    key_in = key

    if key in param_aliases:
        key = param_aliases[key]

    key_short, suffix = remove_path_suffix(key)

    key = key_short if key_short in meta_defaults_default else key  # Try removing training path suffix

    key = key[:-1] if key[:-1] in meta_defaults_default else key  # Try removing training path number

    if (key not in meta_defaults_default) and (substitutions is not None):
        key_short = key
        for substitution in substitutions:
            key_short.replace(substitution[0], substitution[1])
        if key_short in meta_defaults_default:
            key = key_short

    if key not in meta_defaults_default:
        for key_short in reversed(list(meta_defaults_default.keys())):  # reverse as longer names generally come later
        #  in dict defn
            if (key_short in key) and (len(key_short) > 1):  # Avoid matching eg 't' if contains letter 't'
                logger.debug(f'Using {key_short} meta data for {key}')
                key = key_short
                break
    if key not in meta_defaults_default:
        for key_short, value_short in param_aliases.items():
            if (key_short in key) and (len(key_short) > 1):
                key = value_short
                break
        pass

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

def swap_xarray_dim(data_array, new_active_dims, old_active_dims=None, alternative_new_dims=None, raise_on_fail=True):
    """Wrapper for data_array.swap_dims() used to set active coordinate(s) for dimension(s)
    eg. switch between equivalent coordinates like time and frame number.

    Advantages of wrapper:
    - Works out what currently active coordinate to switch so only need to pass new coordinate
    - Doesn't raise an exception if data already has corrector coordinates active
    - Can swap multiple dimension coordinates in one call

    Args:
        data_array      (xr.DataArray) : DataArray for which to swap coordinates
        new_active_dims (str/sequence) : Name(s) of coordinate(s) to set as active
        old_active_dims (str/sequence) : Name(s) of current coordinate(s) to be swapped out with new coords
        alternative_new_dims (str/sequence) : Optional fall back coordinates to switch to if new_active_dims fails

    Returns (xr.DataArray): DataArray with dimension coordinates swapped

    """
    new_active_dims = make_iterable(new_active_dims)

    if isinstance(data_array, np.ndarray):
        logger.warning('Cannot swap dimensions on ndarray')
        return data_array

    for i, new_active_dim in enumerate(new_active_dims):
        if new_active_dim in data_array.dims:
            # Already active, so no change required
            continue

        if old_active_dims is None:
            try:
                old_active_dim = data_array.coords[new_active_dim].dims[0]
            except Exception as e:
                if raise_on_fail:
                    raise
                else:
                    continue  # can't switch dim
        else:
            old_active_dim = make_iterable(old_active_dims)[i]

        try:
            data_array = data_array.swap_dims({old_active_dim: new_active_dim})
        except Exception as e:
            if alternative_new_dims is not None:
                for dim in make_iterable(alternative_new_dims):
                    data_array = swap_xarray_dim(data_array, dim)
            else:
                if raise_on_fail:
                    raise
                else:
                    pass

    return data_array

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
            'units': 'count',
            'units_plot': '',
            'description': 'Camera x pixel coordinate'})
        dataset['y_pix'].attrs.update({
            'long_name': '$y_{pix}$',
            'units': 'count',
            'units_plot': '',
            'description': 'Camera y pixel coordinate'})
        dataset = attach_standard_meta_attrs(dataset, varname=key)
        # TODO: Move to utils/data_structures?
        # TODO: fix latex display of axis labels
        # TODO: use this func in calcam_calibs get_surface_coords
    else:
        raise ValueError(f'Unexpected image data type {data}')
    return dataset


def get_reduce_coord_axis_keep(data, coords_reduce):
    # Get preserved dimension(s) used for xr.apply_ufunc
    if isinstance(data, xr.core.rolling.DataArrayRolling):
        data = data.obj

    ndim = data.values.ndim

    if ndim == 3:
        axes_reduce = [list(data.dims).index(coord_reduce) for coord_reduce in make_iterable(coords_reduce)]
        coord_keep = list(data.dims)
        for coord_reduce in coords_reduce:
            coord_keep.pop(coord_keep.index(coord_reduce))
        coord_keep = coord_keep[0]
        axis_keep = list(data.dims).index(coord_keep)
    elif ndim == 2:
        axes_reduce = list(data.dims).index(coords_reduce)
        coord_keep = list(data.dims)
        coord_keep.pop(data.dims.index(coords_reduce))
        coord_keep = coord_keep[0]
        axis_keep = list(data.dims).index(coord_keep)
    elif ndim == 1:
        coord_keep = None
        axis_keep = None
        axes_reduce = None
    else:
        raise ValueError(f'Too many dimensions: {data}')

    return coord_keep, axis_keep, axes_reduce


def reduce_2d_data_array(data_array, func_reduce, coord_reduce, func_args=(), raise_exceptions=False):
    """Apply a numpy ufunc to reduce 2D DataArray dimensionality to 1D for plotting etc

    Args:
        data_array: Input 2D dataarray
        func_reduce: np ufunc, eg np.mean
        coord_reduce: Coordinate name to apply ufunc across (ie average across time, 't')
        func_args: Any additional args to pass to ufunc eg percentile to pass to np.percentile

    Returns: DataArray with coord_reduce compressed by func_reduce

    """
    named_funcs = dict(mean=np.mean, std=np.std, percentile=np.percentile, mode=utils.mode_simple, max=np.max,
                       min=np.min)
    if isinstance(func_reduce, str):
        func_reduce = named_funcs[func_reduce]

    func_args = make_iterable(func_args)
    coord_keep, axis_keep, axis_reduce = get_reduce_coord_axis_keep(data_array, coord_reduce)

    # TODO: Work out how to input core dims for 3D imput data
    # input_core_dims = ((coord_reduce,), ) + tuple(() for i in np.arange(len(func_args)))
    input_core_dims = (make_iterable(coord_reduce), ) + tuple(() for i in np.arange(len(func_args)))

    try:
        data_array_reduced = xr.apply_ufunc(func_reduce, data_array, *func_args,
                                                    input_core_dims=input_core_dims,
                                            # kwargs=dict(axis=axis_keep)  # Was working with this before?
                                            kwargs=dict(axis=axis_reduce)
                                            )
    except ValueError as e:
        if raise_exceptions:
            raise e
        else:
            logger.warning(f'Cannot reduce data along dim {input_core_dims}: {data_array}')
            out = func_reduce(data_array, *func_args, axis=axis_reduce)
            # out = np.full((len(data_array[coord_keep])), np.nan)
            data_array_reduced = xr.DataArray(out, coords={coord_keep: data_array[coord_keep]})
    except Exception as e:
        raise
    return data_array_reduced


def remove_path_suffix(string, pattern='_path\d+', replacement=''):
    sufix = re.search(pattern, string)
    sufix = sufix.group() if (sufix is not None) else ''
    string_out = re.sub(pattern, replacement, string)
    return string_out, sufix