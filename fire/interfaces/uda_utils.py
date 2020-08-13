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

import pyuda
from fire.misc.utils import increment_figlabel, filter_kwargs, format_str, make_iterable

try:
    # client = pyuda.Client()
    from mast.mast_client import MastClient
    client = MastClient(None)
    # client.server_tmp_dir = ''
except ModuleNotFoundError as e:
    # TODO: Handle missing UDA client gracefully
    raise e

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

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

def get_mastu_wall_coords():
    """
    Best practice advice is to name the file with a .nc file extension.

    Returns:

    """
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
                   kwarg_aliases=None, close=False, **kwargs):
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
        client.put(fn, step_id="create", **create_kwargs)
    except pyuda.UDAException:
        logger.error(f"Failed to create NetCDF file: {fn}")
        raise
    except Exception as e:
        raise
    else:
        logger.info(f'Created uda output netcdf file at {path_fn}')
        if not path_fn.is_file():
            # raise FileNotFoundError(f'UDA output file does not exist: {path_fn}')
            logger.exception(f'UDA output file does not exist: {path_fn}')
    file_id = client.put_file_id
    if close:
        client.put(step_id="close", file_id=file_id, verbose=False)
    return file_id, path_fn

def putdata_update(fn, path=None):
    """Reopen previously created netcdf file

    Args:
        fn:
        path:

    Returns:

    """

    client.put(fn, step_id="update", directory=path, verbose=False)
    file_id = client.put_file_id
    return file_id

def putdata_device(device_name, device_info, attributes=None):
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
    group = f'/devices/{device_name}'
    requried_args = ['id', 'camera_serial_number', 'image_resolution', 'image_range']
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
    except pyuda.UDAException:
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
        putdata_attribute(name, value, group=group)

def putdata_attribute(name, value, group):
    """Write an attribute (misc information) to a file group using UDA putdata interface

    Args:
        name: Name of attribute
        value: Value
        group: Group in file to write attribute to

    Returns: None

    """
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
            client.put(value, name=name, group=group, step_id='attribute')
        except pyuda.UDAException as err:
            logger.warning(f'Failed to write attribute to group "{group}": "{name}"={value}, {err}')
        else:
            logger.debug(f'Successfully wrote attribute to group "{group}": "{name}"={value}')

def putdata_variables_from_datasets(path_data, image_data, path_names,
                                    variable_names_path, variable_names_time, variable_names_image,
                                    existing_dims=None, existing_coords=None):
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
            groups[group] = path_data
            for variable_name in variable_names_path:
                variable_name = path_data[f'{variable_name}_{path_name}']
                group_variables[group].append(variable_name)

    if variable_names_image is not None:
        # Variables with data across the whole image
        group = f'/images'
        groups[group] = image_data
        if 'n' in image_data.dims:
            image_data = image_data.swap_dims({'n': 't'})
        for variable_name in variable_names_image:
            variable = image_data[f'{variable_name}_im']
            group_variables[group].append(variable)

    if variable_names_time is not None:
        # Spatially integrated quantities (eg power to divertor) that are only a function of time
        for path_name in make_iterable(path_names):
            group = f'/{path_name}/integrated'
            groups[group] = path_data
            if 'n' in path_data.dims:
                path_data = path_data.swap_dims({'n': 't'})
            for variable_name in variable_names_time:
                # Pass time data in separate dataset?
                variable = path_data[f'{variable_name}_{path_name}']
                group_variables[group].append(variable)

    # Write dims, coords, data and attributes for variables in each group
    for group in group_variables:
        for variable in group_variables[group]:
            variable_name = variable.name

            # Write dimensions and coords to group in preparation for main variable data
            for coord_name in variable.coords:
                coord = variable[coord_name]

                dim = coord.dims[0]
                coord_class = 'time' if dim == 't' else 'spatial'
                # NOTE: Cannot use same dimension for multiple coordinates?
                if coord_name not in existing_dims[group]:
                    kwargs = {k: coord.attrs[k] for k in optional_dim_attrs if k in coord.attrs}
                    coord_length = len(coord)
                    putdata_dimension(coord_name, coord_length, group=group, **kwargs)
                    existing_dims[group].append(coord_name)

                if coord.name not in existing_coords[group]:
                    kwargs = filter_kwargs(coord.attrs, funcs=putdata_coordinate)
                    values = coord.values
                    putdata_coordinate(coord_name, values=values, coord_class=coord_class, group=group, **kwargs)
                    existing_coords[group].append(coord_name)
                    # TODO: Write additional attributes for coord like axis labels?

            # Write main variable data to group
            dimensions = ','.join(variable.dims)
            kwargs = {k: variable.attrs[k] for k in optional_dim_attrs if k in variable.attrs}
            putdata_variable(variable_name, dimensions, variable.values, group=group, **kwargs)
    logger.info(f'Wrote variables to file: \n{variable_names_image+variable_names_path+variable_names_time}')

def putdata_dimension(dimension_name, dim_length, group='/', **kwargs):
    # TODO: add attributes to dim
    group = format_netcdf_group(group)

    try:
        client.put(dim_length, step_id='dimension', name=dimension_name, group=group, **kwargs)
    except pyuda.UDAException:
        logger.warning(f'Failed to write dimension to group "{group}": "{dimension_name}"')
        raise
    else:
        logger.debug(f'Successfully wrote dimension to group "{group}": "{dimension_name}"')

def putdata_coordinate(coordinate_name, values, coord_class, group='/', units=None, label=None, comment=None):
    group = format_netcdf_group(group)
    units = units_to_uda_conventions(units)
    kwargs = dict(units=units, label=label, comment=comment)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    # TODO: check inputs
    # NOTE name must be name of the associated dimension. This must have been defined in a previous API dimension step
    try:
        client.put(values, step_id='coordinate', name=coordinate_name, group=group, coord_class=coord_class, **kwargs)
    except pyuda.UDAException as e:
        logger.warning(f'Failed to write coordinate to group "{group}": "{coordinate_name}"\n{e}')
        # raise
    else:
        logger.debug(f'Successfully wrote coordinate to group "{group}":  "{coordinate_name}"')

def putdata_variable(variable_name, dimension, values, units=None, label=None, comment=None, group='/',
                     device=None, errors_variable=None, attributes=None, **kwargs):
    group = format_netcdf_group(group)
    units = units_to_uda_conventions(units)

    put_kwargs = dict(units=units, label=label, comment=comment, device=device, errors_variable=errors_variable)
    put_kwargs = {k: v for k, v in put_kwargs.items() if v is not None}
    try:
        client.put(values, step_id='variable', name=variable_name, dimensions=dimension, group=group, **put_kwargs)
    except pyuda.UDAException as e:
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

def putdata_close(file_id=None):
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
    units_allowed = ['s', 'K', 'm', 'count', 'W', 'W/m^2', 'Celsius']
    units_mappings = {}
    if units in units_allowed:
        pass
    elif units in units_mappings:
        units = units_mappings[units]
    else:
        logger.warning(f'Units "{units}" not in FIREs list of UDA compatible units: {units_allowed}')
        units = None
    return units

def putdata_example():
    diag_tag = 'xxx'
    shot = 12345
    pass_number = 0
    status = 1
    conventions = 'Fusion-1.1'
    directory = './'
    # Filename for diagnostics should be <diag_tag><shot_number>.nc
    # Where shotnumber is a 6-digit number with leading zeros (eg. 040255)
    filename = diag_tag + "{:06d}".format(shot) + '.nc'

    try:
        client.put(filename, directory=directory, step_id='create',
                   conventions='Fusion-1.1', shot=shot,
                   data_class='analysed data', title='Example netcdf file',
                   comment='Example MAST-U netcdf file written with putdata in IDL',
                   code='putdata_example.py',
                   pass_number=pass_number, status=1,
                   verbose=True, debug=True)
    except pyuda.UDAException:
        print("<< ERROR >> Failed to create NetCDF file")

    try:
        client.put('device_name', step_id='device',
                   type='type of device', id='#1',
                   serial='123a54', resolution=16,
                   range=[0, 10], channels=32)
    except pyuda.UDAException:
        print('<< ERROR >> Failed to write device to file')

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

    client.put(step_id='close')

if __name__ == '__main__':

    putdata_example()

    shot = 23586
    directory = '/home/tfarley/repos/air/'
    client.put('test.nc', directory=directory, step_id='create',
               conventions='Fusion-1.1', shot=shot,
               data_class='analysed data', title='Example netcdf file',
               comment='Example MAST-U netcdf file written with putdata in IDL',
               code='putdata_example.py',
               pass_number=0, status=1,
               verbose=True, debug=True)

    # putdata_create('./hello.nc', shot=shot, pass_number=0, status=1)

    wall_coords = get_mastu_wall_coords()
    r = client.list(client.ListType.SIGNALS, shot=shot, alias='air')
    signals = [s.signal_name for s in r]
    # signals = ['AIR_QPROFILE_OSP']
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
