#!/usr/bin/env python

"""


Created: 
"""

import logging, os, re
from pprint import pprint
from copy import copy
from collections import defaultdict
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# import pyuda, cpyuda
from fire.physics import physics_parameters
from fire.misc import utils, data_structures
from fire.misc.utils import increment_figlabel, filter_kwargs, format_str, make_iterable
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

from fire.misc.data_structures import meta_defaults_default

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
        "arp_rp radius"]   # (radial position of the reciprocating probe)
    }

# Sentinel for default keyword arguments
module_defaults = object()

def import_pyuda():
    try:
        import pyuda
        client = pyuda.Client()
    except ImportError as e:
        logger.warning(f'Failed to import pyuda. ')
        pyuda, client = False, False
    return pyuda, client

def import_mast_client():
    try:
        from mast import mast_client
        client = mast_client.MastClient(None)
        client.server_tmp_dir = ''
    except ImportError as e:
        logger.warning(f'Failed to import MastClient. ')
        mast_client, client = False, False
    return mast_client, client

def get_uda_client(use_mast_client=False, try_alternative=True):
    try:
        if use_mast_client:
            mast_client, client = import_mast_client()
            uda_module = mast_client
        else:
            pyuda, client = import_pyuda()
            uda_module = pyuda
            # client = pyuda.Client()
    except (ModuleNotFoundError, ImportError) as e:
        # TODO: Handle missing UDA client gracefully
        if try_alternative:
            try:
                uda_module, client = get_uda_client(use_mast_client=~use_mast_client)
            except Exception as e:
                raise e
        else:
            raise e
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
    if (normalise is not False):
        # Normalise data by supplied normalisation factor
        if normalise is True:
            normalise = np.max(np.abs(data_array))
        data_array = data_array / normalise

    return data_array

def uda_signal_obj_to_dataarray(uda_signal_obj, signal, pulse, dims=None, rename_dims='default', meta_defaults='default',
                                raise_exceptions=True):
    if rename_dims == 'default':
        rename_dims = rename_dims_default
    # if meta_defaults == 'default':
    #     meta_defaults = meta_defaults_default

    t = uda_signal_obj.time

    dims = [dim.label for dim in uda_signal_obj.dims]
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
    dims = [d if d not in rename_dims else rename_dims[d] for d in dims]
    data_name = rename_dims.get(uda_signal_obj.label, uda_signal_obj.label)

    # Check for unnamed/'empty string' dimension names
    if any([d.strip() == '' for d in dims]):
        error_message = f'Missing dimension name in {dims} when reading {uda_signal_obj.label}'
        if raise_exceptions:
            raise ValueError(error_message)
        else:
            return ValueError(error_message)

    coords_data = [dim.data for dim in uda_signal_obj.dims]
    data_array = xr.DataArray(uda_signal_obj.data, coords=coords_data, dims=dims)
    data_array.name = data_name
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

def get_uda_meta_dict(uda_obj, key_map, defaults=module_defaults, predefined_values=module_defaults, dim_name=None):
    if dim_name is None:
        dim_name = uda_obj.label
    if (defaults is module_defaults) or (defaults == 'default'):
        defaults = meta_defaults_default.get(dim_name, {})
        if len(defaults) == 0:
            logger.debug(f'No UDA meta defaults for "{dim_name}"')

    if predefined_values is module_defaults:
        predefined_values = meta_refinements_default.get(dim_name, {})

    meta_dict = copy(defaults)
    meta_dict.update(predefined_values)
    for old_key, new_key in key_map.items():
        try:
            value = getattr(uda_obj, old_key)
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
        label_format = axis_label_format_default
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

    if (label not in (False, None)) and ('label' not in plot_kwargs):
        if label is True:
            label = data.attrs.get('symbol', data.name)
        plot_kwargs['label'] = label

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

def get_uda_scheduler_filename(fn_pattern='{diag_tag}{shot:06d}.nc', path=None, kwarg_aliases=None, **kwargs):
    fn = format_str(fn_pattern, kwargs, kwarg_aliases)
    fn = re.sub("^r", 'a', fn)  # Change diagnostic tag from raw to analysed
    if path is not None:
        fn = Path(path) / fn
    return fn

def putdata_create(fn='{diag_tag}{shot:06d}.nc', path='./', shot=None, pass_number=None, status=None,
                   conventions=None, data_class=None, title=None, comment=None, code=None, version=None,
                   xml=None, date=None, time=None, verbose=None,
                   kwarg_aliases=None, close=False, client=None, use_mast_client=False, **kwargs):
    """
    Filename for diagnostics should be <diag_tag><shot_number>.nc
    Where shotnumber is a 6-digit number with leading zeros (eg. air040255)

    Status Description
    -1: Poor Data: Do Not Use (Access to the data via IDAM is blocked
        unless specially requested.)
    0:  Something is Wrong in the Data: Defined either by RO for RAW
        or by Code for ANALYSED
    1:  Either RAW Data Not checked by RO or the Analysis Code
        completed successfully but ANALYSED Data Quality is Unknown
        (Default Value)
    2:  Data has been Checked by the RO - No Known Problems: still
        need ROs permission before publication
    3:  RO has Validated the Data: still subject to clearance

    Args:
        fn:
        path:
        **create_kwargs:

    Returns:

    """
    # import pyuda
    if client is None:
        pyuda_module, client = get_uda_client(use_mast_client=False, try_alternative=False)
        mast_module, client_mast = get_uda_client(use_mast_client=True, try_alternative=False)

    import pyuda
    # Arguments that are different every time
    requried_args = {'shot': shot, 'pass_number': pass_number}#, 'status': status}

    # TODO: Include FIRE version number and git sha
    # TODO: Modify comment message to specifically describe the particular analysis path/camera etc
    # Arguments with standard defaults
    # optional_args = ['directory', 'conventions', 'data_class', 'title', 'comment', 'code', 'version', 'status', 'xml', 'date', 'time', 'verbose']
    passed_kwargs = {'directory': path, 'conventions': conventions, 'data_class': data_class, 'title': title,
                     'comment': comment, 'code': code, 'version': version, 'status': status, 'xml': xml, 'date': date,
                     'time': time, 'verbose': verbose}
    passed_kwargs = {k: v for k, v in passed_kwargs.items() if v is not None}

    create_kwargs = dict(
               directory='./',
               conventions="Fusion-1.1", data_class="analysed data",
               title="Processed infa-red camera data from scheduler for (up to) five IR cameras on MAST-U",
               comment="Temperature, heat flux and other physics quantities derived from IR camera data",
               code="FIRE", version="0.1.0",
               status=1,
               debug=True,  # TODO: Turn off uda debug
               verbose=True)
    create_kwargs.update(passed_kwargs)

    include = list(requried_args.keys()) + list(create_kwargs.keys())
    kwargs_filtered = filter_kwargs(kwargs, include=include, kwarg_aliases=kwarg_aliases)
    create_kwargs.update(kwargs_filtered)

    check_for_required_args(create_kwargs, requried_args.keys(), none_as_missing=True)

    if kwargs:
        fn = get_uda_scheduler_filename(fn, kwarg_aliases=kwarg_aliases, **kwargs)

        create_kwargs['directory'] = os.path.abspath(format_str(create_kwargs['directory'], kwargs, kwarg_aliases))
        create_kwargs['title'] = format_str(create_kwargs['title'], kwargs, kwarg_aliases)
        create_kwargs['comment'] = format_str(create_kwargs['comment'], kwargs, kwarg_aliases)

    path_fn = (Path(create_kwargs['directory'])/fn).resolve()
    # TODO: Delete existing output file?

    try:
        file_id = client.put(fn, step_id="create", **create_kwargs)
    # except pyuda.UDAException:
    #     logger.error(f"Failed to create NetCDF file: {fn}")
    #     raise
    except Exception as e:
        raise
    else:
        logger.debug(f'Created uda output netcdf file handle for {path_fn}')

    try:
        file_id = client._registered_subclients['put'].put_file_id
        # file_id = client_mast.put_file_id
    except AttributeError as e:
        file_id = None
        logger.warning('Failed to return uda netcdf file file_id')
    if close and (file_id is not None):
        client.put(step_id="close", file_id=file_id, verbose=False)
    return file_id, path_fn

def putdata_update(fn, path=None):
    """Reopen previously created netcdf file

    Args:
        fn:
        path:

    Returns:

    """
    uda_module, client = get_uda_client(use_mast_client=False, try_alternative=True)
    client.put(fn, step_id="update", directory=path, verbose=False)
    file_id = client.put_file_id
    return file_id

def putdata_device(device_name, device_info, attributes=None, client=None, use_mast_client=False, file_id=0):
    """
    Add devices (if relevant)
    Devices can be used to record information about
    eg. dataq, diagnostic configuration, model run parameters etc.
    that are relevant to the data stored in the file.
    Only the device name is required, other keywords are optional
    Additional user-defined attributes can be added to devices (see step Attribute later)
    Args:
        device_name:
        device_info:

    Returns:

    """
    # import pyuda
    if client is None:
        uda_module, client = get_uda_client(use_mast_client=use_mast_client, try_alternative=True)

    group = f'/devices/{device_name}'
    requried_args = ['id', 'camera_serial_number', 'detector_resolution', 'image_range']
    check_for_required_args(device_info, requried_args, none_as_missing=True, application='Device data')
    # TODO: Add attributes for wavelength range, manufacuturer, model, lens, wavelength filter, neutral density
    # filter, bit depth
    kwargs_default = dict(type='Infra-red camera', id=None,
                   serial=None, resolution=None,
                   range=[0, 2**14-1], channels=None)
    device_info = copy(device_info)
    kwargs = copy(kwargs_default)
    include = requried_args + list(kwargs_default.keys())
    kws = filter_kwargs(device_info, include=include, remove_from_input=True)
    kwargs.update(kws)
    kwargs = {k: v for k, v in kwargs_default.items() if v is not None}
    try:
        client.put(device_name, step_id='device', **kwargs)
    # except pyuda.UDAException:
    #     raise
    except Exception as e:
        raise
    if attributes is None:
        # Keep all remaining keys by default
        attributes = list(device_info.keys())
        # attributes = [key for key in device_info if key not in kwargs_default]
    # Attach additional attributes to a device that are not default device arguments
    for name in attributes:
        if name not in device_info:
            raise KeyError(f'Missing device attribute: "{name}"')
        value = device_info[name]
        putdata_attribute(name, value, client=client, group=group, file_id=file_id)

def putdata_attribute(name, value, group, client=None, use_mast_client=False, file_id=0):
    """Write an attribute (misc information) to a file group using UDA putdata interface

    Args:
        name: Name of attribute
        value: Value
        group: Group in file to write attribute to

    Returns: None

    """
    # import pyuda
    if client is None:
        uda_module, client = get_uda_client(use_mast_client=use_mast_client, try_alternative=True)

    if isinstance(value, (dict,)):
        # NOTE: Arrays of strings or structures are not supported as single attributes.
        subgroup = f'{group}/{name}'
        for k, v in value.items():
            putdata_attribute(k, v, subgroup)
    elif (isinstance(value, (list, tuple)) and (len(value) > 0) and isinstance(value[0], str)):
        subgroup = f'{group}/{name}'
        for i, v in enumerate(value):
            putdata_attribute(str(i), v, subgroup)
    else:
        try:
            client.put(value, name=name, file_id=file_id, group=group, step_id='attribute')
        # except pyuda.UDAException as err:
        #     logger.warning(f'Failed to write attribute to group "{group}": "{name}"={value}, {err}')
        except Exception as err:
            logger.warning(f'Failed to write attribute to group "{group}": "{name}"={value}, {err}')
        else:
            logger.debug(f'Successfully wrote attribute to group "{group}": "{name}"={value}')

def putdata_variables_from_datasets(path_data, image_data, path_names,
                                    variable_names_path, variable_names_time, variable_names_image, file_id=0,
                                    existing_dims=None, existing_coords=None, client=None, use_mast_client=False):
    # import pyuda
    if existing_dims is None:
        existing_dims = defaultdict(list)
    if existing_coords is None:
        existing_coords = defaultdict(list)
    optional_dim_attrs = ['units', 'label', 'comment']

    # Collect together the names and origin of the variables that are to be written to each file group
    group_variables = defaultdict(list)
    groups = {}
    if variable_names_path is not None:
        if 'n' in path_data.dims:
            path_data = path_data.swap_dims({'n': 't'})

        # Variables with coordinates along an analysis path
        for path_name in make_iterable(path_names):
            # TODO: Use longname for path?
            group = f'/{path_name}'
            path_data = data_structures.swap_xarray_dim(path_data, new_active_dims=f'R_{path_name}',
                                                        raise_on_fail=False)
            groups[group] = path_data
            for variable_name in variable_names_path:
                variable = path_data[f'{variable_name}_{path_name}']
                group_variables[group].append(variable)

    if (variable_names_image is not None) and (len(variable_names_image) > 0):
        # Variables with data across the whole image
        group = f'/images'
        groups[group] = image_data
        if 'n' in image_data.dims:
            image_data = image_data.swap_dims({'n': 't'})
        image_data = data_structures.swap_xarray_dim(image_data, new_active_dims=f'R_{path_name}', raise_on_fail=False)
        for variable_name in variable_names_image:
            variable = image_data[f'{variable_name}_im']
            group_variables[group].append(variable)

    if (variable_names_time is not None) and (len(variable_names_time) > 0):
        # Spatially integrated quantities (eg power to divertor) that are only a function of time
        for path_name in make_iterable(path_names):
            group = f'/{path_name}/stats'
            groups[group] = path_data
            if 'n' in path_data.dims:
                path_data = path_data.swap_dims({'n': 't'})
            # path_data = data_structures.swap_xarray_dim(path_data, new_active_dims=f'R_{path_name}',
            #                                              raise_on_fail=False)
            for variable_name in variable_names_time:
                # Pass time data in separate dataset?
                variable = path_data[f'{variable_name}_{path_name}']
                group_variables[group].append(variable)

    # Write dims, coords, data and attributes for variables in each group
    for group in group_variables:

        for variable in group_variables[group]:
            variable_name = variable.name
            variable = data_structures.swap_xarray_dim(variable, new_active_dims=f'R_{path_name}', raise_on_fail=False)

            # Write dimensions and coords to group in preparation for main variable data
            for coord_name in variable.coords:
                coord = variable[coord_name]

                # path_data = physics_parameters.attach_standard_meta_attrs(path_data, varname=coord_name, replace=True)

                dim_name = coord.dims[0]
                # TODO: switch from using coord_name to dim_name? Cannot use same dimension for multiple coordinates?
                dim_name = coord_name
                coord_class = 'time' if dim_name == 't' else 'spatial'
                # NOTE: Cannot use same dimension for multiple coordinates?
                if dim_name not in existing_dims[group]:
                    kwargs = {k: coord.attrs[k] for k in optional_dim_attrs if k in coord.attrs}
                    coord_length = len(coord)
                    putdata_dimension(dim_name, coord_length, client=client, file_id=file_id, group=group, **kwargs)
                    existing_dims[group].append(dim_name)

                if coord_name not in existing_coords[group]:
                    kwargs = filter_kwargs(coord.attrs, funcs=putdata_coordinate, include=['comment', 'coord_class'],
                                           kwarg_aliases=dict(description='comment'), required='units')
                    values = coord.values
                    putdata_coordinate(coord_name, dim=dim_name, values=values, coord_class=coord_class, client=client,
                                       file_id=file_id, group=group, **kwargs)
                    existing_coords[group].append(coord_name)
                    # TODO: Write additional attributes for coord like axis labels?

            # Write main variable data to group
            dims = [remove_path_suffix(dim)[0] for dim in variable.dims]
            dimensions = ','.join(dims)
            kwargs = {k: variable.attrs[k] for k in optional_dim_attrs if k in variable.attrs}
            putdata_variable(variable_name, dimensions, variable.values, file_id=file_id, group=group,
                             client=client, use_mast_client=use_mast_client, **kwargs)
    logger.info(f'Wrote variables to file: \n   {variable_names_image+variable_names_path+variable_names_time}')

def putdata_dimension(dimension_name, dim_length, client=None, file_id=0, group='/', **kwargs):
    # import pyuda
    # TODO: add attributes to dim
    if client is None:
        uda_module, client = get_uda_client(use_mast_client=False, try_alternative=True)

    group = format_netcdf_group(group)
    dimension_name, path_string = remove_path_suffix(dimension_name)

    kws = filter_kwargs(kwargs, funcs=client.put)

    try:
        out = client.put(dim_length, step_id='dimension', name=dimension_name, group=group, file_id=file_id, **kws)
    # except pyuda.UDAException:
    #     logger.warning(f'Failed to write dimension to group "{group}": "{dimension_name}"')
    #     raise
    except Exception as e:
        logger.warning(f'Failed to write dimension to group "{group}": "{dimension_name}"')
        raise
    else:
        logger.debug(f'Successfully wrote dimension to group "{group}": "{dimension_name}"')

def putdata_coordinate(coordinate_name, dim, values, coord_class, client=None, file_id=0, group='/', units=None,
                       label=None, comment=None):
    # import pyuda
    if client is None:
        uda_module, client = get_uda_client(use_mast_client=False, try_alternative=True)

    group = format_netcdf_group(group)
    dim, path_string = remove_path_suffix(dim)

    units = units_to_uda_conventions(units)
    kwargs = dict(units=units, label=coordinate_name, comment=comment, file_id=file_id)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    # TODO: check inputs
    # NOTE name must be name of the associated dimension. This must have been defined in a previous API dimension step
    try:
        client.put(values, step_id='coordinate', name=dim, group=group, coord_class=coord_class, **kwargs)
    # except pyuda.UDAException as e:
    #     logger.warning(f'Failed to write coordinate to group "{group}": "{coordinate_name}"\n{e}')
        # raise
    except Exception as e:
        logger.warning(f'Failed to write coordinate to group "{group}": "{coordinate_name}"\n{e}')
        raise
    else:
        logger.debug(f'Successfully wrote coordinate to group "{group}":  "{coordinate_name}"')

def putdata_variable(variable_name, dimension, values, units=None, label=None, comment=None, file_id=0,  group='/',
                     device=None, errors_variable=None, attributes=None, client=None, use_mast_client=False, **kwargs):
    # import pyuda
    if client is None:
        uda_module, client = get_uda_client(use_mast_client=use_mast_client, try_alternative=True)

    group = format_netcdf_group(group)
    variable_name, path_string = remove_path_suffix(variable_name)

    units = units_to_uda_conventions(units)

    put_kwargs = dict(units=units, label=label, comment=comment, device=device, errors_variable=errors_variable,
                      file_id=file_id)
    put_kwargs = {k: v for k, v in put_kwargs.items() if v is not None}
    try:
        client.put(values, step_id='variable', name=variable_name, dimensions=dimension, group=group, **put_kwargs)
    # except pyuda.UDAException as e:
    #     logger.warning(f'Failed to write data to group "{group}" for "{variable_name}"')
    #     raise
    except Exception as e:
        logger.warning(f'Failed to write data to group "{group}" for "{variable_name}"')
        raise
    else:
        logger.debug(f'Successfully wrote data to group "{group}" for "{variable_name}"')

    if attributes is None:
        attributes = list(kwargs.keys())

    # Attach additional attributes to a variable that are not default uda arguments
    for name in attributes:
        value = kwargs[name]
        putdata_attribute(name, value, group=group)

# def putdata_group(group, attributes=None):
#     if attributes is not None:
#         # Attach additional attributes to a device that are not default device arguments
#         for name, value in attributes.items():
#             try:
#                 client.put(value, name=name, group=group+f'/', step_id='attribute')
#             except pyuda.UDAException as err:
#                 print(f'<< ERROR >> Failed to write {name} attribute for {device_name}: {err}')
#                 raise

def putdata_close(file_id=None, client=None, use_mast_client=False):
    # import pyuda
    if client is None:
        uda_module, client = get_uda_client(use_mast_client=use_mast_client, try_alternative=True)

    success = False
    kwargs = {}
    if file_id is not None:
        kwargs['file_id'] = file_id
    try:
        client.put(step_id="close", verbose=False, **kwargs)
    except KeyError as e:
        raise
    else:
        success = True
    return success

def format_netcdf_group(group):
    group = f'/{group}' if (group[0] != '/') else group
    return group

# def putdata_error

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
    units_allowed = ['s', 'K', 'm', 'count', 'W', 'W/m^2', 'Celsius', 'degree', 'radian']
    units_mappings = {'MWm$^{-2}$': 'W/m^2',
                      '$^\\circ$C': 'degree',
                      'radians': 'radian'}
                            #  '': 'count'}
    if units in units_allowed:
        pass
    elif units in units_mappings:
        units = units_mappings[units]
    else:
        logger.warning(f'Units "{units}" not in FIREs list of UDA compatible units: {units_allowed}')
        units = None
    return units

def remove_path_suffix(string, pattern='_path\d+', replacement=''):
    sufix = re.search(pattern, string)
    sufix = sufix.group() if (sufix is not None) else ''
    string_out = re.sub(pattern, replacement, string)
    return string_out, sufix

def putdata_example():
    import pyuda

    use_mast_client = False
    # use_mast_client = True
    uda_module, client = get_uda_client(use_mast_client=use_mast_client, try_alternative=True)

    diag_tag = 'xxx'
    shot = 12345
    pass_number = 0
    status = 1
    conventions = 'Fusion-1.1'
    directory = './'
    # Filename for diagnostics should be <diag_tag><shot_number>.nc
    # Where shotnumber is a 6-digit number with leading zeros (eg. 040255)
    filename = diag_tag + "{:06d}".format(shot) + '.nc'
    path_fn = Path(directory) / filename
    if path_fn.exists():
        path_fn.unlink()
    print(f'BEFORE: {path_fn.resolve()} exists: {path_fn.exists()}')

    try:
        file_id = client.put(filename, directory=directory, step_id='create',
                             conventions='Fusion-1.1', shot=shot,
                             data_class='analysed data', title='Example netcdf file',
                             comment='Example MAST-U netcdf file written with putdata in IDL',
                             code='putdata_example.py',
                             pass_number=pass_number, status=1,
                             verbose=True, debug=True)
    except pyuda.UDAException as e:
        print("<< ERROR >> Failed to create NetCDF file")
    except Exception as e:
        raise
    print(f'After CREATE: {path_fn} exists: {path_fn.exists()}')

    try:
        client.put('device_name', step_id='device',
                   type='type of device', id='#1',
                   serial='123a54', resolution=16,
                   range=[0, 10], channels=32)
    except pyuda.UDAException:
        print('<< ERROR >> Failed to write device to file')
    print(f'After DEVICE: {path_fn} exists: {path_fn.exists()}')

    try:
        client.put(100, step_id='dimension', name='time1', group='/xxx')
    except pyuda.UDAException:
        print('<< ERROR >> Failed to write dimension time1 to file')

    # /xxx/time1 : 0 -> 0.99 sec
    try:
        client.put(np.arange(100, dtype=np.float32) * 0.01, step_id='coordinate', name='time1', group='/xxx',
                   units='s', label='time', comment='time axis with 10 ms resolution',
                   coord_class='time')
    except pyuda.UDAException:
        print('<< ERROR >> Failed to write time1 coordinate data')

    # Example errors /xxx/group_b/error_variable1
    try:
        client.put(np.arange(100, dtype=np.float32) * 20.0, step_id='variable', name='error_variable1',
                   group='/xxx/group_b',
                   dimensions='time1', units='kA', label='Errors for variable1',
                   comment='Estimate of errors for variable 1 of 10 % ')
    except pyuda.UDAException:
        print('<< ERROR >> Failed to write error data error_variable1')

    try:
        client.put(np.arange(100, dtype=np.float32) * 200.0, step_id='variable', name='variable1', group='/xxx/group_b',
                   dimensions='time1', units='kA', errors='error_variable1',
                   label='variable 1', comment='example variable 1 with associated errors error_variable1')
    except pyuda.UDAException:
        print('<< ERROR >> Failed to write data for variable1')

    # Attach additional attributes to a device
    try:
        client.put('Extra information about device_name', step_id='attribute',
                   group='/devices/device_name', name='notes')
    except pyuda.UDAException as err:
        print('<< ERROR >> Failed to write notes attribute {}'.format(err))

    print(f'Before CLOSE: {path_fn} exists: {path_fn.exists()}')

    client.put(step_id='close')

    print(f'After CLOSE: {path_fn} exists: {path_fn.exists()}')

    print(f'use_mast_client={use_mast_client}')
    if path_fn.exists():
        path_fn.unlink()
        print('SUCCESS: Deleted output file')
    else:
        print('FAILURE: Output file not produced')

def uda_get_signal_example():
    uda_module, client = get_uda_client(use_mast_client=False, try_alternative=True)

    ip = client.get('ip', 18299)
    rir = client.get_images('rir', 29976, first_frame=400, last_frame=900)
    print(f'ip: {ip}')
    print(f'rir: {rir}')

if __name__ == '__main__':
    uda_module, client = get_uda_client(use_mast_client=False, try_alternative=True)

    putdata_example()
    exit()
    uda_get_signal_example()

    pulse = 23586
    directory = '/home/tfarley/repos/air/'
    client.put('test.nc', directory=directory, step_id='create',
               conventions='Fusion-1.1', shot=pulse,
               data_class='analysed data', title='Example netcdf file',
               comment='Example MAST-U netcdf file written with putdata in IDL',
               code='putdata_example.py',
               pass_number=0, status=1,
               verbose=True, debug=True)

    # putdata_create('./hello.nc', shot=shot, pass_number=0, status=1)

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
