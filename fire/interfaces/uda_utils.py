#!/usr/bin/env python

"""


Created: 
"""

import logging, re
from pprint import pprint
from copy import copy
from collections import defaultdict
from pathlib import Path
from functools import lru_cache

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# import pyuda, cpyuda
from fire.misc.data_structures import variables_meta_defaults
from fire.misc.utils import increment_figlabel, format_str
from fire.plotting import plot_tools

# use_mast_client = True
# try:
#     if use_mast_client:
#         from mast.mast_client import MastClient
#         client = MastClient(None)
#         client.server_tmp_dir = ''
#     else:
#         client = pyuda.Client()
# except ModuleNotFoundError as e:
#     # TODO: Handle missing UDA client gracefully
#     raise e

logging.basicConfig()  # TODO: remove
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # TODO: remove

"""UDA documentation at: https://users.mastu.ukaea.uk/data-access-and-tools/pyuda
UDA code at: https://git.ccfe.ac.uk/MAST-U/UDA
"""

# Improved names for signals (replacing uda name meta data) used for eg axis labels
signal_aliases = {
    'analysed_ext_rog_TF': r'$I_p$',  # /AMC/ROGEXT/TF
    '/xim/da/hm10/t': r'$D_{\alpha,\paralell}$',  # /xim/da/hm10/t
    '/xim/da/hm10/r': r'$D_{\alpha,\perp}$',  # /xim/da/hm10/t
    'V': r'$D_{\alpha}$',  # /xim/da/hm10/t  # TODO: tmp
}
# TODO: Add simplified alias for labelling axis when plotting similar signals eg different channels of same signal

# TODO: Move module defaults to json files
# Ordered names of dimensions for different signals, as many uda signals have missing dimension names. These names
# should match the UDA naming conventions eg using 'Time' not 't'
signal_dims_lookup = {
    'AIR_QPROFILE_OSP': ['Time', 'Radius'],  # 'Time', 'S'?
    'AIR_QPROFILE_ISP': ['Time', 'Radius'],
    'AIR_QPROFILE_ISP': ['Time', 'Radius'],
    'AIR_QPROFILE_ISP_ELM': ['Time', 'Radius'],
    'AIR_QPROFILE_OSP': ['Time', 'Radius'],
    'AIR_QPROFILE_OSP_ELM': ['Time', 'Radius'],
    'AIR_RCOORD_ISP': ['Time', 'Radius'],
    'AIR_RCOORD_OSP': ['Time', 'Radius'],
    'AIR_TPROFILE_ISP': ['Time', 'Radius'],
    'AIR_TPROFILE_OSP': ['Time', 'Radius']
}

rename_dims_default = {'Time': 't',
                       'Radius': 'R',
                       'IR true coord': 'S',  # S?
                       'Poloidal coord': 'S',
                       'Temperature': 'T',
                       'Temperature profiles': 'T',
                       'IR power profiles': 'q'}


# Replacements for existing uda meta data eg add latex formatting
meta_refinements_default = {
    'IR power profiles': {'units': '$kW m^{-2}$'},  # TODO: check kW vs MW
    'Temperature profiles': {'units': '$^\circ$C'},
}

axis_label_format_default = '{name} [{units}]'
axis_label_key_combos_default = [['symbol', 'units'], ['label', 'units'], ['name', 'units']]

signal_abbreviations = {
    'Ip': "amc_plasma current",
    'ne': "ESM_NE_BAR",  # (electron line averaged density)
    'ne2': "ane_density",  # Gives lower density - what is this?
    'Pnbi': "anb_tot_sum_power",  # Total NBI power
    'Pohm': "esm_pphi",  # Ohmic heating power (noisy due to derivatives!)
    'Ploss': "esm_p_loss",  # Total power crossing the separatrix
    'q95': "efm_q_95",  # q95
    'q0': "efm_q_axis",  # q0
    'q952': "EFM_Q95",  # (q95)
    'Da': "ada_dalpha integrated",
    # 'Da-mp': 'ph/s/cm2/sr',
    'sXray': 'xsx/tcam/1',
    'Bphi': 'efm_bphi_rmag',
    'zmag': "efm_magnetic_axis_z",  # Hight of magnetic axis (used to distinguish LSND and DND)
    'dn_edge': "ADG_density_gradient",
    'Bvac': "EFM_BVAC_VAL",  # (vacuum field at R=0.66m)
    'LPr': "arp_rp radius",  # (radial position of the reciprocating probe)
    'LWIR1_trig': 'xpx/clock/lwir-1'
}
signal_sets = {
    'set1': [
        "amc_plasma current",
        "ESM_NE_BAR",  # (electron line averaged density)
        "ane_density",  # Gives lower density - what is this?
        "anb_tot_sum_power",  # Total NBI power
        "esm_pphi",  # Ohmic heating power (noisy due to derivatives!)
        "esm_p_loss",  # Total power crossing the separatrix
        "efm_q_95",  # q95
        "efm_q_axis",  # q0
        "EFM_Q95",  # (q95)
        "ada_dalpha integrated",
        'xsx/tcam/1',  # soft xray 1
        'efm_bphi_rmag',
        "efm_magnetic_axis_z",  # Hight of magnetic axis (used to distinguish LSND and DND)
        "ADG_density_gradient",
        "EFM_BVAC_VAL",  # (vacuum field at R=0.66m)
        "arp_rp radius"],   # (radial position of the reciprocating probe)
    'session_leader_set': [  # Sigals Jack Lovell uses when session leading:
        '/ane/density',
        '/epm/output/separatrixgeometry/rmidplanein',
        '/epm/output/separatrixgeometry/rmidplaneout',
        '/xdc/ai/cpu1/plasma_current',
        '/xdc/ai/cpu1/rogext_p1',
        '/xdc/ai/cpu1/rogext_p6l',
        '/xdc/reconstruction/s/seg01_r',  #  (LEMUR outer radius)
        '/xim/da/hm10/r',
        '/xim/da/hm10/t',
        '/xzc/zcon/zip']
    }


# Sentinel for default keyword arguments
module_defaults = object()

derived_signals = ['ne_bar', 'greenwald_density', 'greenwald_frac']

def get_greewald_density(shot):
    from mastu_exhaust_analysis.read_efit import read_uda
    efit_data = read_uda(shot)

    signal = 'greenwald_density'
    meta = dict(shot=shot, pulse=shot, signal=signal, label=signal)

    data_array = signal_arrays_to_dataarray(data=efit_data['ngw'], dims=('t',), coords={'t': efit_data['t']},
                                            signal=signal, name=signal, meta_dict=meta)
    return data_array

def get_ne_bar(shot, efit_data=None):
    """Get average volumetric density along interferometer chord"""
    from mastu_exhaust_analysis.calc_ne_bar import calc_ne_bar
    data = calc_ne_bar(shot=shot, efit_data=efit_data)

    signal='ne_bar'
    meta = dict(shot=shot, pulse=shot, signal=signal, label=signal, units='m$^{-3}$')

    data_array = signal_arrays_to_dataarray(data=data['data'], dims=('t',), coords={'t': data['t']},
                                            signal=signal, name=signal, meta_dict=meta)
    return data_array

def get_greenwald_density_frac(shot, efit_data=None):
    from mastu_exhaust_analysis.calc_ne_bar import calc_ne_bar
    data = calc_ne_bar(shot, efit_data=efit_data)

    greenwald_frac = data['greenwald_fraction']
    signal = 'greenwald_frac'
    meta = dict(shot=shot, pulse=shot, signal=signal, label=signal, units='')

    data_array = signal_arrays_to_dataarray(data=greenwald_frac, dims=('t',), coords={'t': data['t']},
                                            signal=signal, name=signal, meta_dict=meta)

    # greenwald_density = greenwald_density.where(~np.isinf(greenwald_density), drop=True).dropna(dim='time')
    # greenwald_density = greenwald_density.interp(coords={'time': ne_bar['time']}, method='cubic', )
    # greenwald_frac = ne_bar / greenwald_density

    return data_array

def get_derived_signal(signal, shot, **kwargs):
    derived_signals_funcs = dict(ne_bar=get_ne_bar, greenwald_density=get_greewald_density,
                                 greenwald_frac=get_greenwald_density_frac)

    data_array = derived_signals_funcs[signal](shot, **kwargs)
    return data_array

# Return same client instance from previous call
@lru_cache(maxsize=2, typed=False)
def import_pyuda():
    try:
        import pyuda
        client = pyuda.Client()
    except ImportError as e:
        logger.warning(f'Failed to import pyuda. ')
        pyuda, client = False, False
    return pyuda, client

# Return same client instance from previous call
@lru_cache(maxsize=2, typed=False)
def import_mast_client():
    try:
        from mast import mast_client
        client = mast_client.MastClient(None)
        client.server_tmp_dir = ''
    except ImportError as e:
        logger.warning(f'Failed to import MastClient. ')
        mast_client, client = False, False
    return mast_client, client

def get_uda_client(use_mast_client=False, try_alternative=True, server="uda2.hpc.l", port=56565):
    try:
        if use_mast_client:
            mast_client, client = import_mast_client()
            uda_module = mast_client
        else:
            pyuda, client = import_pyuda()
            uda_module = pyuda
            # UDA2 (new office network server). UDA3 = remote access server
            # See other servers at https://users.mastu.ukaea.uk/sites/default/files/uploads/UDA_data_access.pdf

            # pyuda.Client.server = server
            # pyuda.Client.port = port

    except (ModuleNotFoundError, ImportError) as e:
        # TODO: Handle missing UDA client gracefully
        if try_alternative:
            try:
                uda_module, client = get_uda_client(use_mast_client=~use_mast_client)
            except Exception as e:
                raise e
        else:
            raise e
    # client.set_property('get_meta', True)
    return uda_module, client

def filter_uda_signals(signal_string, pulse=23586):
    """Return list of signal names that contain the supplied signal_string

    Args:
        signal_string: String element of signal name (case insensitive) e.g. "air"
        pulse: pulse/shot number

    Returns: List of signal names

    """
    # r = client.list(client.ListType.SIGNALS, shot=shot, alias='air')
    uda_module, client = get_uda_client(use_mast_client=False, try_alternative=True)
    r = client.list_signals(shot=pulse, alias=signal_string)
    signals = [s.signal_name for s in r]

    return signals

def read_uda_signal(signal, pulse, raise_exceptions=True, log_exceptions=True, use_mast_client=False,
                    try_alternative=True, **kwargs):
    uda_module, client = get_uda_client(use_mast_client=use_mast_client, try_alternative=True)
    import cpyuda

    signal = signal_abbreviations.get(signal, signal)

    try:
        data = client.get(signal, pulse, **kwargs)
    except (cpyuda.ServerException, AttributeError) as e:
        if try_alternative:
            logger.debug(f'Trying alternative uda client: use_mast_client={not use_mast_client}')
            data = read_uda_signal(signal, pulse, raise_exceptions=raise_exceptions, log_exceptions=log_exceptions,
                                   use_mast_client=(not use_mast_client), try_alternative=False, **kwargs)
        else:
            error_message = f'Failed to read UDA signal for: "{signal}", {pulse}, {kwargs}. "{e}"'
            exception = RuntimeError(error_message)
            if raise_exceptions:
                raise exception
            else:
                if log_exceptions:
                    logger.warning(error_message)
                data = exception

    return data

def read_uda_signal_to_dataarray(signal, pulse, dims=None, rename_dims='default', meta_defaults='default',
                                 use_mast_client=False, normalise=False, raise_exceptions=True) -> xr.DataArray:
    uda_data_obj = read_uda_signal(signal, pulse, use_mast_client=use_mast_client, raise_exceptions=raise_exceptions)

    if isinstance(uda_data_obj, Exception) and (not raise_exceptions):
        return uda_data_obj

    data_array = uda_signal_obj_to_dataarray(uda_data_obj, signal, pulse, dims=dims, rename_dims=rename_dims,
                                                 meta_defaults=meta_defaults, raise_exceptions=raise_exceptions)

    data_array = normalise_signal(data_array, normalise=normalise)

    return data_array

def normalise_signal(data, normalise=True):
    if (normalise is not False):
        # Normalise data by supplied normalisation factor
        if normalise is True:
            normalise = np.max(np.abs(data))
        data = data / normalise

    return data

def signal_arrays_to_dataarray(data, dims, coords, signal, name, rename_dims='default', meta_dict=(),
                               meta_defaults='default', raise_exceptions=True):
    if rename_dims == 'default':
        rename_dims = rename_dims_default

    meta_dict = dict(meta_dict)

    dims_lookup = signal_dims_lookup.get(signal)

    for i, dim in enumerate(dims):
        if dims_lookup is not None:
            if dim.strip() == '':
                logger.warning(f'Replacing empty UDA "{signal}" coordinate[{i}] name "{dim}" with "{dims_lookup[i]}"')
                dims[i] = dims_lookup[i]
            elif dims[i] != dims_lookup[i]:
                error_message = f'UDA and lookup dim names do not match. UDA: {dims}, lookup: {dims_lookup}'
                if raise_exceptions:
                    raise ValueError(error_message)
                else:
                    return ValueError(error_message)
        if dims[i].strip() == '':
            error_message = f'UDA signal contains empty dim name and signal missing from signal_dims_lookup: {dims}'
            if raise_exceptions:
                raise ValueError(error_message)
            else:
                return ValueError(error_message)

    # Rename dims eg 'Time' -> 't'
    dims = [d if d.capitalize() not in rename_dims else rename_dims[d.capitalize()] for d in dims]

    # Check for unnamed/'empty string' dimension names
    if any([d.strip() == '' for d in dims]):
        error_message = f'Missing dimension name in {dims} for {name}'
        if raise_exceptions:
            raise ValueError(error_message)
        else:
            return ValueError(error_message)

    data_array = xr.DataArray(data, coords=coords, dims=dims, name=name)
    data_array.attrs.update(meta_dict)

    # TODO: Lookup default meta data
    meta_keys = {'label': 'name', 'units': 'units'}  # , 'description': 'description'}
    meta_dict = get_uda_meta_dict(data_array, dim_name=name, key_map=meta_keys, defaults=meta_defaults)
    data_array.attrs.update(meta_dict)

    for coord_name, dim_name in zip(data_array.coords, dims):
        coord_obj = data_array.coords[coord_name]
        # TODO: Lookup default meta data
        meta_dict = get_uda_meta_dict(coord_obj, key_map=meta_keys, dim_name=coord_name)
        coord_obj.attrs.update(meta_dict)

    return data_array

def uda_signal_obj_to_dataarray(uda_signal_obj, signal, pulse, dims=None, rename_dims='default', meta_defaults='default',
                                raise_exceptions=True):
    if rename_dims == 'default':
        rename_dims = rename_dims_default

    t = uda_signal_obj.time
    dims = [dim.label for dim in uda_signal_obj.dims]
    coords_data = [dim.data for dim in uda_signal_obj.dims]
    data_name = rename_dims.get(uda_signal_obj.label, uda_signal_obj.label)

    data_array = signal_arrays_to_dataarray(uda_signal_obj.data, dims=dims, coords=coords_data,
                                            signal=signal, name=data_name,  meta_dict=(), rename_dims=rename_dims,
                                            meta_defaults=meta_defaults, raise_exceptions=raise_exceptions)

    meta_keys = {'label': 'name', 'units': 'units', 'description': 'description'}
    meta_dict = get_uda_meta_dict(uda_signal_obj, dim_name=data_name,
                                  key_map=meta_keys, defaults=meta_defaults)
    data_array.attrs.update(meta_dict)

    data_array.attrs['signal'] = signal
    data_array.attrs['shot'] = pulse
    data_array.attrs['pulse'] = pulse
    data_array.attrs['uda_signal'] = uda_signal_obj

    meta_keys = {'label': 'name', 'units': 'units'}  # , 'description': 'description'}
    for coord_name, dim_obj in zip(data_array.coords, uda_signal_obj.dims):
        coord = data_array.coords[coord_name]
        meta_dict = get_uda_meta_dict(dim_obj, key_map=meta_keys, dim_name=coord_name)
        coord.attrs.update(meta_dict)

    return data_array

def list_uda_signals_for_shot(pulse, filter_string=None):
    signals = filter_uda_signals(filter_string, pulse=pulse)
    return signals

def get_uda_meta_dict(uda_obj, key_map, defaults=module_defaults, predefined_values=module_defaults, dim_name=None,
                      signal=None):
    if isinstance(uda_obj, xr.DataArray):
        get_attr_func = uda_obj.attrs.get
    else:
        def get_attr_func(key):
            getattr(uda_obj, key)

    # label = getattr(uda_obj, 'label', uda_obj.attrs.get('label'))
    label = get_attr_func('label')

    if (defaults is module_defaults) or (defaults == 'default'):
        for name in [dim_name, signal, label]:
            defaults = variables_meta_defaults.get(name, {})
            if len(defaults) == 0 and (isinstance(name, str)):
                defaults = variables_meta_defaults.get(name.lower(), {})
            if len(defaults) > 0:
                break
        else:
            logger.debug(f'No UDA meta defaults for "{dim_name}"')

    if predefined_values is module_defaults:
        predefined_values = meta_refinements_default.get(dim_name, {})

    meta_dict = copy(defaults)
    meta_dict.update(predefined_values)
    for old_key, new_key in key_map.items():
        try:
            value = get_attr_func(old_key)
            if value is None:
                raise AttributeError
        except AttributeError as e:
            logger.debug(f'Object {uda_obj} does not have attr "{old_key}"')
        else:
            if value.strip() != '':
                meta_dict[new_key] = value
        if new_key in predefined_values:
            meta_dict[new_key] = predefined_values[new_key]

    # TODO: Split into separate function to update meta_dict with aliases
    axis_label = gen_axis_label_from_meta_data(meta_dict)
    if axis_label is not None:
        meta_dict['axis_label'] = axis_label

    # Add attrs used by xarray for automatic axis labeling (precedence: long_name, standard_name,  DataArray.name)
    if 'symbol' in meta_dict:
        meta_dict['standard_name'] = meta_dict.get('symbol')
    # if 'name' in meta_dict:
    #     meta_dict['long_name'] = meta_dict.get('name')
    return meta_dict

def gen_axis_label_from_meta_data(meta_data, label_format=None, label_key_combos=None):
    if label_format is None:
        if meta_data.get('units', None) not in ['', None]:
            label_format = axis_label_format_default
        else:
            label_format = '{name}'

    if label_key_combos is None:
        label_key_combos = axis_label_key_combos_default

    # Try to generate an axis label with available keys
    for name, units in label_key_combos:
        axis_label = None
        try:
            name = meta_data[name]
            if name in signal_aliases:
                name = signal_aliases[name]
                meta_data['name'] = name
                meta_data['symbol'] = name
            axis_label = label_format.format(name=name, units=meta_data[units])
        except KeyError as e:
            pass
        else:
            break
    return axis_label

def describe_uda_signal(signal, pulse=23586, raise_exceptions=True, log_exceptions=True, **kwargs):
    uda_obj = read_uda_signal(signal, pulse, raise_exceptions=raise_exceptions, log_exceptions=log_exceptions, **kwargs)
    if isinstance(uda_obj, Exception):
        return uda_obj

    t = uda_obj.time
    out = {}
    out['data.description'] = uda_obj.description
    out['data.label'] = uda_obj.label
    out['data.units'] = uda_obj.units
    out['data.shape'] = uda_obj.data.shape
    out['data.size'] = uda_obj.data.size
    out['t.shape'] = t.data.shape
    out['data.dims'] = uda_obj.dims
    out['data.rank'] = uda_obj.rank
    out['data.meta'] = uda_obj.meta
    out['data_obj'] = uda_obj
    out['x_label'] = f'{t.label}'
    out['x_units'] = f'{t.units}'
    out['x_axis_label'] = f'{t.label} [{t.units}]'

    z = None
    if uda_obj.rank == 0:
        raise NotImplementedError
    elif uda_obj.rank == 1:
        y = uda_obj
    elif uda_obj.rank == 2:
        y = uda_obj.dims[1]
        z = uda_obj
    out['y_label'] = f'{y.label}'
    out['y_units'] = f'{y.units}'
    out['y_axis_label'] = f'{y.label} [{y.units}]'

    if (z is not None):
        out['z_label'] = f'{z.label}'
        out['z_units'] = f'{z.units}'
        out['z_axis_label'] = f'{z.label} [{z.units}]'
    else:
        out['z_label'] = None
        out['z_units'] = None
        out['z_axis_label'] = None
    # out['data.errors'] = data.errors
    out['dim_ranges'] = [[dim.data.min(), dim.data.max()] for dim in uda_obj.dims]
    # out['dir(data)'] = dir(data)
    return out

def sort_uda_signals_by_ndims(signals, sort_by='dim_names', pulse=23586,
                              raise_exceptions=False, log_exceptions=True, **kwargs):

    out = defaultdict(list)

    for signal in signals:
        info = describe_uda_signal(signal, pulse, raise_exceptions=raise_exceptions, log_exceptions=log_exceptions,
                                   **kwargs)
        if isinstance(info, Exception):
            out['error_reading_signal'].append(signal)
        else:
            if sort_by == 'dim_names':
                key = tuple(dim.label for dim in info['data.dims'])
                if len(key) == 1:
                    key = key[0]
                if info['data.size'] == 1:
                    key = 'scalar:'+key
            else:
                key = info['data.rank']
            out[key].append(signal)

    return out

def plot_uda_signal(signal, pulse, dims=None, ax=None, verbose=False, **kwargs):
    data_array = read_uda_signal_to_dataarray(signal, pulse, dims=dims)

    plot_uda_dataarray(data_array, ax=ax, plot_kwargs=kwargs)

def get_default_plot_kwargs_for_style(style):
    # TODO: Move defaults to json file and add user input option
    if style == 'line':
        # plot_kwargs = dict(marker='o', markersize=3)
        plot_kwargs = dict()
    elif style == 'pcolormesh':
        plot_kwargs = dict(robust=True, center=False, cmap='coolwarm')  # 'plasma', 'RdBu', 'RdYlBu', 'bwr'
    elif style == 'imshow':
        plot_kwargs = dict(robust=True, center=False)
    elif style == 'contourf':
        plot_kwargs = dict(robust=True, center=False)
    else:
        plot_kwargs = {}
    return plot_kwargs

def plot_uda_dataarray(data, xdim=None, style=None, plot_kwargs=None, ax=None, label=True, show=True,
                       tight_layout=True):
    # TODO: Generalise to non-UDA data and enable swapping of axis coordinates eg n->t, R->s etc (see FIRE 2d debug
    # plots)
    if ax is None:
        num = increment_figlabel(f'{data.attrs["signal"]}:{data.attrs["shot"]}', i=2, suffix=' ({i})')
        fig, ax = plt.subplots(num=num)

    if (len(data.dims) > 1) and (len(data['t']) == 1):
        data = data.sel(t=data['t'][0])

    if style is None:
        if (len(data.dims) == 1):
            style = 'line'
        elif len(data.dims) == 2:
            style = 'pcolormesh'

    plot_kwargs_default = get_default_plot_kwargs_for_style(style)
    if plot_kwargs is None:
        plot_kwargs = plot_kwargs_default
    else:
        plot_kwargs_default.update(plot_kwargs)
        plot_kwargs = plot_kwargs_default

    name = data.attrs.get('symbol', data.name)
    shot = data.attrs.get('shot', data.attrs.get('pulse'))
    if (label not in (False, None)) and ('label' not in plot_kwargs):
        if label is True:
            label = name
        plot_kwargs['label'] = label
    if isinstance(plot_kwargs.get('label'), str):
        plot_kwargs['label'] = plot_kwargs.get('label').format(symbol=name, name=name, shot=shot, pulse=shot)

    if xdim is None:
        xdim = list(data.coords)[0]
    if (data.ndim == 1) or (style in ('line',)):
        ydim = data.name
        y = data
    else:
        ydim = list(data.coords)[1]
        y = data[ydim]

    x = data.coords[xdim]

    # ax.plot(t.data, data.data)
    if style is not None:
        plot_method = getattr(data.plot, style)
    else:
        plot_method = data.plot

    fig_artist = plot_method(ax=ax, **plot_kwargs)

    if 'axis_label' in data.attrs:
        ax.set_ylabel(data.attrs['axis_label'])

    # ax.set_xlabel(x.attrs['axis_label'])
    # ax.set_ylabel(y.attrs['axis_label'])
    # if data.ndim > 1:
    #     fig_artist.colorbar.ax.set_ylabel(data.attrs['axis_label'])#, rotation=270)

    plot_tools.show_if(show=show, tight_layout=tight_layout)

    return ax, fig_artist

def get_signal_as_dataarray(signal, pulse, dims=None, normalise=False):
    """Read UDA signal or derived signal"""
    if signal in derived_signals:
        data_array = get_derived_signal(signal, pulse)
        normalise_signal(data_array, normalise=normalise)
    else:
        data_array = read_uda_signal_to_dataarray(signal, pulse, dims=dims, normalise=normalise, raise_exceptions=False)

    return data_array

def plot_uda_signal(signal, pulse, dims=None, ax=None, normalise=False, show=True, verbose=False, **kwargs):

    data_array = read_uda_signal_to_dataarray(signal, pulse, dims=dims, normalise=normalise, raise_exceptions=False)

    if isinstance(data_array, Exception):
        return ax, data_array, None

    if verbose:
        logger.info(f'{signal}, {pulse}: min={data_array.min().values:0.4g}, mean={data_array.mean().values:0.4g}, '
                    f'99%={np.percentile(data_array.values,99):0.4g}, max={data_array.max().values:0.4g}')

    # pprint(data_array)
    ax, artist = plot_uda_dataarray(data_array, ax=ax, show=show, plot_kwargs=kwargs)

    return ax, data_array, artist
    # TODO: Add function for plotting slices through data
    # data_slices = data_array.T.sel(t=[0, 0.1, 0.2, 0.242, 0.3], method='nearest')
    # plot_uda_dataarray(data_slices, style='line', plot_kwargs=dict(x='S'), show=False)
    # plot_uda_dataarray(data_array.T, show=False)
    # data = read_ir_uda('ip', 18299)
    # plt.show()

def plot_uda_signals_individually(pulse=23586, signals='all', diagnostic_alias='air', min_rank=1, verbose=True):
    """

    Args:
        pulse: pulse/shot number
        signals: List of uda signal names. If "all" then will plot all signals containing diagnostic_alias
        diagnostic_alias: string to filter uda singals by eg "air" for MWIR camera
        min_rank: Used to skip plotting low dimensionality data >=1 to skip scalar data, >=2 to skip 1D data

    Returns: None

    """

    if signals == 'all':
        signals = filter_uda_signals(diagnostic_alias, pulse=pulse)

    signals_by_ndims = sort_uda_signals_by_ndims(signals)

    pprint(signals)
    pprint(signals_by_ndims)

    for signal in signals:
        dims = signal_dims_lookup.get(signal)
        info = describe_uda_signal(signal, pulse, raise_exceptions=False)
        if isinstance(info, Exception):
            continue
        if (info['data.size'] <= 1) and (min_rank > 0):
            print(f'Skipping plot for scalar data: {signal}')
            continue
        if (info['data.rank'] <= 1) and (min_rank > 1):
            print(f'Skipping plot for 1D data: {signal}')
            continue

        print(f'\n\nPlotting {info["data.rank"]}D signal "{signal}" for pulse {pulse} with dims {dims}')
        # pprint(info)

        plot_uda_signal(signal, pulse=pulse, verbose=verbose)

def get_mastu_wall_coords():
    """
    Best practice advice is to name the file with a .nc file extension.

    Returns:

    """
    uda_module, client = get_uda_client(use_mast_client=False, try_alternative=True)
    coords = client.geometry("/limiter/efit", 50000, no_cal=True)
    print(coords)
    return coords

def get_uda_scheduler_filename(fn_pattern='{diag_tag_analysed}{shot:06d}.nc', path=None, kwarg_aliases=None, **kwargs):
    fn = format_str(fn_pattern, kwargs, kwarg_aliases)
    fn = get_analysed_diagnostic_tag(fn)  # Change diagnostic tag from raw to analysed
    if path is not None:
        fn = Path(path) / fn
    return fn

def change_tag_case(tag, upper=False, lower=False):
    if upper:
        tag = tag.upper()
    if lower:
        tag = tag.upper()
    return tag

def get_analysed_diagnostic_tag(diag_tag, upper=False, lower=False):
    # TODO: allow caps input
    analysed_tag = re.sub("^r", 'a', diag_tag)  # Change diagnostic tag from raw to analysed
    analysed_tag = change_tag_case(analysed_tag, upper=upper, lower=lower)
    return analysed_tag

def get_raw_diagnostic_tag(diag_tag, upper=False, lower=False):
    raw_tag = re.sub("^a", 'r', diag_tag)  # Change diagnostic tag from raw to analysed
    raw_tag = change_tag_case(raw_tag, upper=upper, lower=lower)
    return raw_tag

def get_diag_tag_variants(diag_tag):
    diag_tag = diag_tag.lower()
    tags = {}
    tags['diag_tag_raw'] = get_raw_diagnostic_tag(diag_tag)
    tags['diag_tag_raw_upper'] = get_raw_diagnostic_tag(diag_tag, upper=True)
    tags['diag_tag_analysed'] = get_analysed_diagnostic_tag(diag_tag)
    tags['diag_tag_analysed_upper'] = get_analysed_diagnostic_tag(diag_tag, upper=True)
    return tags

def check_for_required_args(kwargs, required_args, none_as_missing=True, application=None):
    # Record of missing required args
    missing_args = []
    application = '' if application is None else f'{application}: '
    for key in required_args:
        if key not in kwargs:
            missing_args.append(key)
        elif none_as_missing and (kwargs[key] is None):
            missing_args.append(key)
    if len(missing_args) > 0:
        raise KeyError(f'{application}Missing required argument(s): {", ".join(missing_args)}')

def units_to_uda_conventions(units):
    # TODO: Get list of supported units from UDA

    units_allowed = ['s', 'K', 'm', 'count', 'W', 'W/m^2', 'Celsius', 'degree', 'radian', 'J', 'MW', 'MJ']
    units_mappings = {
                      'MWm$^{-2}$': 'W/m^2',
                      'MW': 'W',
                      'MJ': 'J',
                      '$^\\circ$C': 'degree',
                      'radians': 'radian'}
                            #  '': 'count'}
    if units is None:
        logger.warning(f'Units "{units}" not in FIREs list of UDA compatible units: {units_allowed}')
        return units

    units_simple = units.replace('$', '')

    if units in units_allowed:
        pass
    elif units in units_mappings:
        units = units_mappings[units]
    elif units_simple in units_allowed:
        units = units_simple
    else:
        logger.warning(f'Units "{units}" not in FIREs list of UDA compatible units: {units_allowed}')
        units = None
    return units


def latest_uda_shot_number():
    import pyuda
    client = pyuda.Client()
    shot_no_last = int(client.latest_shot())
    return shot_no_last

def uda_get_signal_example():
    uda_module, client = get_uda_client(use_mast_client=False, try_alternative=True)

    ip = client.get('ip', 18299)
    rir = client.get_images('rir', 29976, first_frame=400, last_frame=900)
    print(f'ip: {ip}')
    print(f'rir: {rir}')

if __name__ == '__main__':
    uda_module, client = get_uda_client(use_mast_client=False, try_alternative=True)

    uda_get_signal_example()

    # wall_coords = get_mastu_wall_coords()

    signals = filter_uda_signals('air', pulse=pulse)
    # signals = ['AIR_QPROFILE_OSP']
    pprint(signals)
    signals_by_ndims = sort_uda_signals_by_ndims(signals)
    pprint(signals_by_ndims)
    # signals = [signals[0]]
    print(f'pulse={pulse}')
    for signal in signals:
        dims = signal_dims_lookup.get(signal)
        info = describe_uda_signal(signal, pulse, raise_exceptions=False)
        if isinstance(info, Exception):
            continue
        if info['data.size'] <= 1:
            print(f'Skipping plot for scalar data: {signal}')
            continue
        if info['data.rank'] <= 1:
            print(f'Skipping plot for 1D data: {signal}')
            continue

        print(f'\n\nPlotting signal "{signal}" with dims {dims}')
        pprint(info)
        data = read_uda_signal(signal, pulse, raise_exceptions=False)
        # data.plot()
        data_array = read_uda_signal_to_dataarray(signal, pulse, dims=dims, raise_exceptions=False)
        if isinstance(data_array, Exception):
            continue
        pprint(data_array)

        plot_uda_signal(signal, pulse, dims=dims, show=False)

        # data_slices = data_array.T.sel(t=[0, 0.1, 0.2, 0.242, 0.3], method='nearest')
        # plot_uda_dataarray(data_slices, style='line', plot_kwargs=dict(x='S'), show=False)
        # plot_uda_dataarray(data_array.T, show=False)
        # data = read_ir_uda('ip', 18299)
        plt.show()

    # pprint(r)
    pass
