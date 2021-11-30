#!/usr/bin/env python

"""


Created: 
"""

import logging
import os
from collections import defaultdict
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from fire.misc import data_structures
from fire.misc.data_structures import remove_path_suffix
from fire.misc.utils import filter_kwargs, format_str, make_iterable, safe_len
from fire.physics import physics_parameters

from fire.interfaces.uda_utils import (get_uda_client, check_for_required_args, get_uda_scheduler_filename, logger,
    get_analysed_diagnostic_tag, get_raw_diagnostic_tag, units_to_uda_conventions)

logger = logging.getLogger(__name__)

def putdata_create(fn='{diag_tag_analysed}{shot:06d}.nc', path='./', shot=None, pass_number=None, status=None,
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
        # mast_module, client_mast = get_uda_client(use_mast_client=True, try_alternative=False)

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
        # create_kwargs['directory'] = create_kwargs['directory'].format(**kwargs)
        fn = get_uda_scheduler_filename(fn, kwarg_aliases=kwarg_aliases, **kwargs)

        create_kwargs['directory'] = format_str(create_kwargs['directory'], kwargs, kwarg_aliases)
        create_kwargs['title'] = format_str(create_kwargs['title'], kwargs, kwarg_aliases)
        create_kwargs['comment'] = format_str(create_kwargs['comment'], kwargs, kwarg_aliases)

    create_kwargs['directory'] = str(Path(create_kwargs['directory']).expanduser().resolve())
    path_fn = (Path(create_kwargs['directory']) / fn)
    # TODO: Delete existing output file?

    try:
        file_id = client.put(fn, step_id="create", **create_kwargs)

        # client.put('ait044852.nc', directory='/home/tfarley/repos/air', step_id='create', title='Test #21',
        #            data_class='analysed data', pass_number=0, status=1, conventions='Fusion - 1.0', debug=True, verbose=True, shot=12345)

        # except pyuda.UDAException:
    #     logger.error(f"Failed to create NetCDF file: {fn}")
    #     raise
    except Exception as e:
        raise e
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

    group = f'devices/{device_name}/'  #
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


def putdata_variables_from_datasets(path_data, image_data, path_names, diag_tag,
                                    variable_names_path, variable_names_time, variable_names_image, file_id=0,
                                    existing_dims=None, existing_coords=None, client=None, use_mast_client=False):
    # import pyuda
    if existing_dims is None:
        existing_dims = defaultdict(list)
    if existing_coords is None:
        existing_coords = defaultdict(list)
    optional_dim_attrs = ['units', 'label', 'comment']

    diag_tag_analysed = get_analysed_diagnostic_tag(diag_tag)
    diag_tag_raw = get_raw_diagnostic_tag(diag_tag)

    # Collect together the names and origin of the variables that are to be written to each file group
    group_variables = defaultdict(list)
    groups = {}
    if variable_names_path is not None:
        if 'n' in path_data.dims:
            path_data = path_data.swap_dims({'n': 't'})

        # Variables with coordinates along an analysis path
        for path_name in make_iterable(path_names):
            # TODO: Use longname for path?
            group = f'/{diag_tag_analysed}/{path_name}'
            path_data = data_structures.swap_xarray_dim(path_data, new_active_dims=f'R_{path_name}',
                                                        raise_on_fail=False)
            groups[group] = path_data
            for variable_name in variable_names_path:
                variable = path_data[f'{variable_name}_{path_name}']
                group_variables[group].append(variable)

    if (variable_names_image is not None) and (len(variable_names_image) > 0):
        # Variables with data across the whole image
        group = f'/{diag_tag_analysed}/{path_name}/images'
        groups[group] = image_data
        if 'n' in image_data.dims:
            image_data = image_data.swap_dims({'n': 't'})
        # image_data = data_structures.swap_xarray_dim(image_data, new_active_dims=f'R_{path_name}', raise_on_fail=False)
        for variable_name in variable_names_image:
            variable = image_data[f'{variable_name}_im']
            group_variables[group].append(variable)

    if (variable_names_time is not None) and (len(variable_names_time) > 0):
        # Spatially integrated quantities (eg power to divertor) that are only a function of time
        for path_name in make_iterable(path_names):
            group = f'/{diag_tag_analysed}/{path_name}'  # /stats'
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
        group_data = groups[group]
        for variable in group_variables[group]:
            variable_name = variable.name
            # variable = data_structures.swap_xarray_dim(variable, new_active_dims=f'R_{path_name}', raise_on_fail=False)

            # Write dimensions and coords to group in preparation for main variable data
            for coord_name in variable.coords:

                # Make sure coord has required attrs like units etc
                group_data = physics_parameters.attach_standard_meta_attrs(group_data, varname=coord_name, replace=True)
                coord = group_data[coord_name]

                # dim_name = coord.dims[0]
                # TODO: switch from using coord_name to dim_name? Cannot use same dimension for multiple coordinates?
                dim_name = coord_name
                coord_class = 'time' if (dim_name in ('t', 'n')) else 'spatial'
                # NOTE: Cannot use same dimension for multiple coordinates?
                if dim_name not in existing_dims[group]:
                    kwargs = {k: coord.attrs[k] for k in optional_dim_attrs if k in coord.attrs}
                    coord_length = safe_len(coord)
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

    if label is None:
        label = coordinate_name

    units = units_to_uda_conventions(units)
    kwargs = dict(units=units, label=label, comment=comment, file_id=file_id)
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
    if label is None:
        label = variable_name

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
        raise e
    except Exception as e:
        raise e
    else:
        success = True
    return success


def format_netcdf_group(group):
    group = f'/{group}' if (group[0] != '/') else group
    return group


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



# def putdata_group(group, attributes=None):
#     if attributes is not None:
#         # Attach additional attributes to a device that are not default device arguments
#         for name, value in attributes.items():
#             try:
#                 client.put(value, name=name, group=group+f'/', step_id='attribute')
#             except pyuda.UDAException as err:
#                 print(f'<< ERROR >> Failed to write {name} attribute for {device_name}: {err}')
#                 raise

# def putdata_error

if __name__ == '__main__':
    uda_module, client = get_uda_client(use_mast_client=False, try_alternative=True)

    putdata_example()

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