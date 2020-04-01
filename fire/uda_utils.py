#!/usr/bin/env python

"""


Created: 
"""

import logging
from pprint import pprint
from copy import copy

import xarray as xr
import matplotlib.pyplot as plt

import pyuda
from fire.utils import increment_figlabel

client = pyuda.Client()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# TODO: Move module defaults to json files
signal_dims = {
    'AIR_QPROFILE_OSP': ['Time', 'S']
}

rename_dims_default = {'Time': 't'}
meta_defaults_default = {
    't': {'units': 's', 'symbol': '$t$'},
    'S': {'label': 'Divertor tile S coordinate', 'units': 'm', 'symbol': '$S$'},
    'IR power profiles': {'symbol': '$q_\perp$'},
}
meta_refinements_default = {
    'IR power profiles': {'units': '$kW m^{-2}$'},
}

axis_label_format_default = '{name} [{units}]'
axis_label_key_combos_default = [['symbol', 'units'], ['label', 'units'], ['name', 'units']]

# Sentinel for default keyword arguments
module_defaults = object()

def read_uda_signal(signal, pulse):
    data = client.get(signal, pulse)
    return data

def read_uda_signal_to_dataarrays(signal, pulse, dims=None, rename_dims='default', meta_defaults='default'):
    if rename_dims == 'default':
        rename_dims = rename_dims_default
    if meta_defaults == 'default':
        meta_defaults = meta_defaults_default

    data = read_uda_signal(signal, pulse)
    t = data.time

    if dims is None:
        dims = [dim.label for dim in data.dims]
    # Rename dims
    dims = [d if d not in rename_dims else rename_dims[d] for d in dims]
    if any([d.strip() == '' for d in dims]):
        raise ValueError(f'Missing dimension name in {dims} when reading {signal}:{pulse}')

    coords = [dim.data for dim in data.dims]
    data_array = xr.DataArray(data.data, coords=coords, dims=dims)
    meta_keys = {'label': 'name', 'units': 'units', 'description': 'description'}
    meta_dict = get_uda_meta_dict(data, key_map=meta_keys)
    data_array.attrs.update(meta_dict)

    data_array.attrs['signal'] = signal
    data_array.attrs['shot'] = shot
    data_array.attrs['pulse'] = shot
    data_array.attrs['uda_signal'] = data

    meta_keys = {'label': 'name', 'units': 'units'}  # , 'description': 'description'}
    for coord_name, dim_obj in zip(data_array.coords, data.dims):
        coord = data_array.coords[coord_name]
        meta_dict = get_uda_meta_dict(dim_obj, key_map=meta_keys, dim_name=coord_name)
        coord.attrs.update(meta_dict)

    return data_array

def get_uda_meta_dict(obj, key_map, defaults=module_defaults, predefined_values=module_defaults, dim_name=None):
    if dim_name is None:
        dim_name = obj.label
    if defaults is module_defaults:
        defaults = meta_defaults_default.get(dim_name, {})
        if len(defaults) == 0:
            logger.debug(f'No UDA meta defaults for "{dim_name}"')
    if predefined_values is module_defaults:
        predefined_values = meta_refinements_default.get(dim_name, {})
    meta_dict = copy(defaults)
    meta_dict.update(predefined_values)
    for old_key, new_key in key_map.items():
        try:
            value = getattr(obj, old_key)
        except AttributeError as e:
            logger.debug(f'Object {obj} does not have attr "{old_key}"')
        else:
            if value.strip() != '':
                meta_dict[new_key] = value
        if new_key in predefined_values:
            meta_dict[new_key] = predefined_values[new_key]
    axis_label = gen_axis_label_from_meta_data(meta_dict)
    if axis_label is not None:
        meta_dict['axis_label'] = axis_label
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
            axis_label = label_format.format(name=meta_data[name], units=meta_data[units])
        except KeyError as e:
            pass
        else:
            break
    return axis_label

def describe_uda_signal(signal, pulse=23586):
    data = read_uda_signal(signal, pulse)
    t = data.time
    out = {}
    out['xlabel'] = f'{t.label} [{t.units}]'
    out['ylabel'] = f'{data.label} [{data.units}]'
    out['data.description'] = data.description
    out['data.shape'] = data.data.shape
    out['t.shape'] = t.data.shape
    out['data.dims'] = data.dims
    out['data.rank'] = data.rank
    out['data.meta'] = data.meta

    # out['data.errors'] = data.errors
    out['dim_ranges'] = [[dim.data.min(), dim.data.max()] for dim in data.dims]
    # out['dir(data)'] = dir(data)
    return out

def plot_uda_signal(signal, pulse, dims=None, **kwargs):
    data_array = read_uda_signal_to_dataarrays(signal, pulse, dims=dims)
    plot_uda_dataarray(data_array, **kwargs)

def get_default_plot_kwargs_for_style(style):
    # TODO: Move defaults to json file and add user input option
    if style == 'line':
        plot_kwargs = dict(marker='o', markersize=4)
    else:
        plot_kwargs = {}
    return plot_kwargs

def plot_uda_dataarray(data, xdim=None, style=None, plot_kwargs=None, ax=None, show=True, tight_layout=True):
    if ax is None:
        num = increment_figlabel(f'{data.attrs["signal"]}:{data.attrs["shot"]}', i=2, suffix=' ({i})')
        fig, ax = plt.subplots(num=num)
    if style is None:
        if len(data) == 1:
            style = 'line'
    plot_kwargs_default = get_default_plot_kwargs_for_style(style)
    if plot_kwargs is None:
        plot_kwargs = plot_kwargs_default
    else:
        plot_kwargs_default.update(plot_kwargs)
        plot_kwargs = plot_kwargs_default

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
    plot_method(**plot_kwargs)

    ax.set_xlabel(x.attrs['axis_label'])
    ax.set_ylabel(y.attrs['axis_label'])

    if tight_layout:
        plt.tight_layout()
    if show:
        plt.show()

if __name__ == '__main__':
    shot = 23586
    r = client.list(pyuda.ListType.SIGNALS, shot=shot, alias='air')
    # signals = ['AIR_QPROFILE_OSP']
    signals = [s.signal_name for s in r]
    pprint(signals)
    # signals = [signals[0]]
    print(f'pulse={shot}')
    for signal in signals:
        dims = signal_dims.get(signal)
        print(f'\n\nPlotting signal "{signal}" with dims {dims}')
        info = describe_uda_signal(signal, shot)
        pprint(info)
        data = read_uda_signal(signal, shot)
        # data.plot()
        data_array = read_uda_signal_to_dataarrays(signal, shot, dims=dims)
        pprint(data_array)
        data_slices = data_array.T.sel(t=[0, 0.1, 0.2, 0.242, 0.3], method='nearest')
        plot_uda_signal(signal, shot, dims=dims, style='line', show=False)
        # plot_uda_dataarray(data_slices, style='line', plot_kwargs=dict(x='S'), show=False)
        # plot_uda_dataarray(data_array.T, show=False)
        # data = read_ir_uda('ip', 18299)
        plt.show()

    # pprint(r)
    pass
