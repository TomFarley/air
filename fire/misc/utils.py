# -*- coding: future_fstrings -*-
#!/usr/bin/env python

"""Miscelanious utility functions

Created: 11-10-19
"""

import logging, inspect, os, re
from typing import Union, Iterable, Tuple, List, Optional, Any, Sequence, Callable
from pathlib import Path
from copy import copy

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

PathList = Sequence[Union[Path, str]]

def movie_data_to_dataarray(frame_data, frame_times, frame_nos=None, meta_data=None, name='frame_data'):
    """Return frame data in xarray.DataArray object

    Args:
        frame_data  : Array of camera digit level data with dimensions [t, y, x]
        frame_times : Array of frame times
        frame_nos   : Array of frame numbers

    Returns: DataArray of movie data

    """
    if frame_nos is None:
        frame_nos = np.arange(frame_data.shape[0])
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

def update_call_args(user_defaults, pulse, camera, machine):
    """Replace 'None' values with user's preassigned default values

    :param user_defaults: Dict of user's default settings
    :param pulse: Shot/pulse number or string name for synthetic movie data
    :param camera: Name of camera to analyse (unique name of camera or diagnostic code)
    :param machine: Tokamak that the data originates from
    :return:
    """
    if pulse is None:
        pulse = user_defaults['pulse']
    if camera is None:
        camera = user_defaults['camera']
    if machine is None:
        machine = user_defaults['machine']
    machine = sanitise_machine_name(machine)
    camera = sanitise_camera_name(camera)
    return pulse, camera, machine

def sanitise_machine_name(machine):
    machine_out = machine.lower().replace('-', '_')
    return machine_out

def sanitise_camera_name(camera):
    camera_out = camera.lower().replace('-', '_')
    return camera_out

def make_iterable(obj: Any, ndarray: bool=False,
                  cast_to: Optional[type]=None,
                  cast_dict: Optional=None,
                  # cast_dict: Optional[dict[type,type]]=None,
                  nest_types: Optional=None) -> Iterable:
                  # nest_types: Optional[Sequence[type]]=None) -> Iterable:
    """Return itterable, wrapping scalars and strings when requried.

    If object is a scalar nest it in a list so it can be iterated over.
    If ndarray is True, the object will be returned as an array (note avoids scalar ndarrays).

    Args:
        obj         : Object to ensure is iterable
        ndarray     : Return as a non-scalar np.ndarray
        cast_to     : Output will be cast to this type
        cast_dict   : dict linking input types to the types they should be cast to
        nest_types  : Sequence of types that should still be nested (eg dict)

    Returns:

    """
    if not hasattr(obj, '__iter__') or isinstance(obj, str):
        obj = [obj]
    if (nest_types is not None) and isinstance(obj, nest_types):
        obj = [obj]
    if (cast_dict is not None) and (type(obj) in cast_dict):
        obj = cast_dict[type(obj)](obj)
    if ndarray:
        obj = np.array(obj)
    if (cast_to is not None):
        if isinstance(cast_to, (type, Callable)):
            if cast_to == np.ndarray:
                obj = np.array(obj)
            else:
                obj = cast_to(obj)  # cast to new type eg list
        else:
            raise TypeError(f'Invalid cast type: {cast_to}')
    return obj

def is_in(items, collection):
    """Return boolean mask, True for each item in items that is present in collection"""
    items = make_iterable(items)
    collection = make_iterable(collection)
    out = pd.Series(items).isin(collection).values
    return out

def get_traceback_location(level=0, format='{module_name}:{func_name}:{line_no} '):
    """Returns an informative prefix for verbose Debug output messages"""
    module = module_name(level=level)
    func = func_name(level=level)
    line = line_no(level=level)
    return format.format(module_name=module, func_name=func, line_no=line)

def is_possible_filename(fn, ext_whitelist=('py', 'txt', 'png', 'p', 'npz', 'csv',), ext_blacklist=(),
                         ext_max_length=3):
    """Return True if 'fn' is a valid filename else False.

    Return True if 'fn' is a valid filename (even if it and its parent directory do not exist)
    To return True, fn must contain a file extension that satisfies:
        - Not in blacklist of extensions
        - May be in whitelist of extensions
        - Else has an extension with length <= ext_max_length
    """
    fn = str(fn)
    ext_whitelist = ['.' + ext for ext in ext_whitelist]
    if os.path.isfile(fn):
        return True
    elif os.path.isdir(fn):
        return False

    ext = os.path.splitext(fn)[1]
    l_ext = len(ext) - 1
    if ext in ext_whitelist:
        return True
    elif ext in ext_blacklist:
        return False

    if (l_ext > 0) and (l_ext <= ext_max_length):
        return True
    else:
        return False

def mkdir(dirs, start_dir=None, depth=None, accept_files=True, info=None, verbose=1):
    """ Create a set of directories, provided they branch of from an existing starting directory. This helps prevent
    erroneous directory creation. Checks if each directory exists and makes it if necessary. Alternatively, if a depth
    is supplied only the last <depth> levels of directories will be created i.e. the path <depth> levels above must
    pre-exist.
    Inputs:
        dirs 			- Directory path
        start_dir       - Path from which all new directories must branch
        depth           - Maximum levels of directories what will be created for each path in <dirs>
        info            - String to write to DIR_INFO.txt file detailing purpose of directory etc
        verbatim = 0	- True:  print whether dir was created,
                          False: print nothing,
                          0:     print only if dir was created
    """
    from pathlib import Path
    # raise NotImplementedError('Broken!')
    if start_dir is not None:
        start_dir = os.path.expanduser(str(start_dir))
        if isinstance(start_dir, Path):
            start_dir = str(start_dir)
        start_dir = os.path.abspath(start_dir)
        if not os.path.isdir(start_dir):
            print('Directories {} were not created as start directory {} does not exist.'.format(dirs, start_dir))
            return 1

    if isinstance(dirs, Path):
        dirs = str(dirs)
    if isinstance(dirs, (str)):  # Nest single string in list for loop
        dirs = [dirs]
    # import pdb; pdb.set_trace()
    for d in dirs:
        if isinstance(d, Path):
            d = str(d)
        d = os.path.abspath(os.path.expanduser(d))
        if is_possible_filename(d):
            if accept_files:
                d = os.path.dirname(d)
            else:
                raise ValueError('mkdir was passed a file path, not a directory: {}'.format(d))
        if depth is not None:
            depth = np.abs(depth)
            d_up = d
            for i in np.arange(depth):  # walk up directory by given depth
                d_up = os.path.dirname(d_up)
            if not os.path.isdir(d_up):
                logger.info('Directory {} was not created as start directory {} (depth={}) does not exist.'.format(
                    d, d_up, depth))
                continue
        if not os.path.isdir(d):  # Only create if it doesn't already exist
            if (start_dir is not None) and (start_dir not in d):  # Check dir stems from start_dir
                if verbose > 0:
                    logger.info('Directory {} was not created as does not start at {} .'.format(dirs,
                                                                                          os.path.relpath(start_dir)))
                continue
            try:
                os.makedirs(d)
                if verbose > 0:
                    logger.info('Created directory: {}   ({})'.format(d, get_traceback_location(level=2)))
                if info:  # Write file describing purpose of directory etc
                    with open(os.path.join(d, 'DIR_INFO.txt'), 'w') as f:
                        f.write(info)
            except FileExistsError as e:
                logger.warning('Directory already created in parallel thread/process: {}'.format(e))
        else:
            if verbose > 1:
                logger.info('Directory "' + d + '" already exists')
    return 0

def dirs_exist(paths: Iterable[Union[str, Path]], path_kws: Optional[dict]=None
               ) -> Tuple[List[Path], List[str], List[str]]:
    paths = make_iterable(paths)
    if path_kws is None:
        path_kws = {}

    paths_exist = []
    paths_raw_exist = []
    paths_raw_not_exist = []

    for path_raw in paths:
        # Insert missing info into format strings
        path_raw = str(path_raw)
        try:
            path = path_raw.format(**path_kws)
        except KeyError as e:
                raise ValueError(f'Cannot locate file without value for "{e.args[0]}": "{path_raw}", {path_kws}"')
        try:
            path = Path(path).expanduser().resolve(strict=False)
        except RuntimeError as e:
            if "Can't determine home directory" in str(e):
                pass
                # continue
            else:
                raise e
        except (FileNotFoundError, TypeError) as e:
            path = Path(path).expanduser()
            logger.warning(e)
        if path.is_dir():
            paths_exist.append(path)
            paths_raw_exist.append(path_raw)
        else:
            paths_raw_not_exist.append(path_raw)
    return paths_exist, paths_raw_exist, paths_raw_not_exist

def locate_file(paths: Iterable[Union[str, Path]], fns: Iterable[str],
                path_kws: Optional[dict]=None, fn_kws: Optional[dict]=None,
                return_raw_path: bool=False, return_raw_fn: bool=False,
                raise_: bool=True, verbose: Union[bool, int]=False) \
                -> Union[Tuple[Path, str], Tuple[str, str], Tuple[None, None]]:
    """Return path to file given number of possible paths

    Args:
        paths               : Possible paths where files could be located
        fns                 : Possible filename formats for requested file i.e. the filename may be slightly different
                              depending on the location it is found in
        path_kws            : Values to substitute into path format strings
        fn_kws              : Values to substitute into filename format strings
        return_raw_path     : Return path without 'path_kws' substitutions
        return_raw_fn       : Return filename without 'fn_kws' substitutions
        raise_              : Raise an exception if the file is not located
        verbose             : Log whether or not the file was located

    Returns: (path, filename) / (None, None)

    """
    # TODO: detect multiple occurences/possible paths
    # TODO: Allow regular expresssions
    fns = make_iterable(fns)
    if fn_kws is None:
        fn_kws = {}

    located = False
    paths_dont_exist = []
    paths_exist, paths_raw_exist, paths_raw_not_exist = dirs_exist(paths, path_kws=path_kws)
    for path, path_raw in zip(paths_exist, paths_raw_exist):
        for fn_raw in fns:
            try:
                fn = str(fn_raw).format(**fn_kws)
            except KeyError as e:
                raise ValueError(f'Cannot locate file without value for "{e.args[0]}": "{fn_raw}", {fn_kws}"')
            except IndexError as e:
                raise e
            except ValueError as e:
                logger.debug(f'Incompatible type for value in filename format string "{fn_raw}": \n{fn_kws}\n{e}')
                continue
            fn_path = path / fn
            if fn_path.is_file():
                located = True
                path_out = path_raw if return_raw_path else path
                fn_out = fn_raw if return_raw_fn else fn
                if verbose >= 2:
                    logging.info('Located "{}" in {}'.format(fn_out, path_out))
                break
            else:
                paths_dont_exist.append(fn_path)
        if located:
            break
    else:
        # File not located
        message = (f'Failed to locate file with formats: {fns} in paths: \n"{paths}" \n' 
                   f'with fn_kws: \n{fn_kws} \n' 
                   f'File possibilities checked that do not exist: \n{", ".join([str(p) for p in paths_dont_exist])}.')
        if raise_:
            raise FileNotFoundError(message)
        if verbose:
            logger.warning(message)
        path_out, fn_out = None, None
    return path_out, fn_out

def locate_files(paths: Iterable[Union[str, Path]], fns: Iterable[str],
                 path_kws: Optional[dict]=None, fn_kws: Optional[dict]=None,
                 return_raw_path: bool=False, return_raw_fn: bool=False,
                 raise_: bool=False, verbose: Union[bool, int]=False) \
                -> List[Path]:
    """Return paths to files that exist, for all combinations on input paths and filenames

    Args:
        paths               : Possible paths where files could be located
        fns                 : Possible filename formats for requested file i.e. the filename may be slightly different
                              depending on the location it is found in
        path_kws            : Values to substitute into path format strings
        fn_kws              : Values to substitute into filename format strings
        return_raw_path     : Return path without 'path_kws' substitutions
        return_raw_fn       : Return filename without 'fn_kws' substitutions
        raise_              : Raise an exception if the file is not located
        verbose             : Log whether or not the file was located

    Returns: List of located path objects

    """
    files_located = []
    paths = make_iterable(paths)
    fns = make_iterable(fns)
    for path in paths:
        for fn in fns:
            p, f = locate_file(path, fn, path_kws=path_kws, fn_kws=fn_kws, return_raw_path=return_raw_path,
                               return_raw_fn=return_raw_fn, raise_=raise_, verbose=verbose)
            if isinstance(f, (Path, str)):
                files_located.append(Path(p)/f)
    return files_located

def join_path_fn(path: Union[Path, str], fn: str):
    """Return path object resulting from joining the path and fn inputs

    Args:
        path    : File path
        fn      : Filename

    Returns: Path object to file

    """
    return Path(path) / fn

def convert_dataframe_values_to_python_types(df, col_subset=None, allow_strings=True, list_delimiters=',',
                                             strip_chars=' '):
    import pandas as pd
    if col_subset is None:
        col_subset = list(df.columns)

    for col in col_subset:
        if not isinstance(df[col].dtype, (object, pd.StringDtype, str)):
            continue
        column_dict = df[col].to_dict()
        column_dict = convert_dict_values_to_python_types(column_dict, allow_strings=allow_strings,
                                                          list_delimiters=list_delimiters, strip_chars=strip_chars)
        for key, value in column_dict.items():
            if isinstance(value, (list, tuple)):
                df = df.astype({col: object})
            df.at[key, col] = value  # Use 'at' rather than 'loc' to allow assigning lists to a cell

    return df

def convert_dict_values_to_python_types(dictionary, keys_subset=None, allow_strings=True, list_delimiters=',',
                                        strip_chars=' '):
    if keys_subset is None:
        keys_subset = list(dictionary.keys())

    for key in keys_subset:
        value = dictionary[key]
        if isinstance(value, str):
            dictionary[key] = convert_string_to_python_type(value, allow_strings=allow_strings,
                                                            list_delimiters=list_delimiters, strip_chars=strip_chars)

    return dictionary

def list_repr_to_list(string_list, allow_strings=True, list_delimiters=','):
    """Convert the string representation of a list to a python list

    Args:
        string_list: String representing list
        allow_strings: Whether to allow string elements in the list, else raise an error (bool)

    Returns: python list object

    """
    # TODO: extend for tuple
    string = string_list.strip('[ ]')

    for delim in make_iterable(list_delimiters):
        elements = string.split(delim)
        if len(elements) > 1:
            break

    for i, element in enumerate(elements):
        elements[i] = convert_string_to_python_type(element, allow_strings=allow_strings)
    return elements

def convert_string_to_python_type(string, allow_strings=True, strip_chars=' ', list_delimiters=','):
    """Convert a repr string to it's python type

    Supported types: int, float, None, list, (str)
    """

    if re.match("\w*\[.*\]\w*", string):
        return list_repr_to_list(string, allow_strings=allow_strings, list_delimiters=list_delimiters)

    type_dict = {'none': None, 'true': True, 'false': False}
    for key, value in type_dict.items():
        if string.lower() == key:
            return value

    try:
        out = int(string)
    except ValueError as e:
        pass
    else:
        return out

    try:
        out = float(string)
    except ValueError as e:
        pass
    else:
        return out

    if not allow_strings:
        raise ValueError(f'Failed to convert string "{string}" to a python type')
    else:
        if strip_chars is not None:
            string = string.strip(strip_chars)
        return string


def filter_kwargs(kwargs, funcs=None, include=(), exclude=(), required=None, kwarg_aliases=None,
                  extract_func_dict=True, remove_from_input=False):
    """Return filtered dict of kwargs that match input call signature for supplied function.

    Args:
        kwargs           : Dict of kwargs to be filtered
        funcs            : Function(s) whose signatures should be used to filter the kwargs
        include          : List of keys to include regardless of function signature
        exclude          : List of keys to exclude regardless of function signature
        required         : List of keys that must be located and returned else an error is raised
        kwarg_aliases    : Dict mapping key names in kwargs to names in func signatures to enable matches
        extract_func_dict: If kwargs contains a dict under key '<func_name>_args' return its contents (+ filtered kwargs)
        remove_from_input: Remove filtered kwargs from original input kwargs dict

    Returns: Dictionary of keyword arguments that are compatible with the supplied function's call signature

    """
    #TODO: Include positional arguments!
    kwargs_out = {}
    keep = []  # list of all keys to keep (passing filter)
    sig_matches = []  # keys matching func signatures
    func_name_args = []  # Keys for dists of kwargs specific to supplied function(s)

    if funcs is not None:
        for f in make_iterable(funcs):
            # Add arguments for each function to list of arguments to keep
            if isinstance(f, type):
                # If a class look at it's __init__ method
                sig_matches += list(inspect.signature(f.__init__).parameters.keys())
            else:
                sig_matches += list(inspect.signature(f).parameters.keys())
            func_name_arg = '{name}_args'.format(name=f.__name__)
            if func_name_arg in kwargs:
                func_name_args += [func_name_arg]

    for key, value in kwargs.items():
        key_names = [key]
        if (kwarg_aliases is not None) and (key in kwarg_aliases):
            key_names += make_iterable(kwarg_aliases[key])
        for k in key_names:
            if (((k in sig_matches) and (k not in exclude)) or (k in include)):
                if (k not in kwargs_out) or (kwargs_out[k] == value):
                    keep.append(key)
                    kwargs_out[k] = value
                else:
                    raise ValueError(f'Passed conflicting kwargs values for key "{k}": {kwargs_out[k]}, {value}')

    if extract_func_dict:
        # Extract values from dicts under keys named '<func_name>_args'
        for key in func_name_args:
            kwargs_out.update(kwargs_out[key])

    if required is not None:
        missing = []
        for key in required:
            if key not in kwargs_out:
                missing.append(key)
        if missing:
            raise ValueError(f'Missing required input keyword arguments {missing} from kwargs: {kwargs}')

    if remove_from_input:
        # Remove key value pairs from kwargs that were transferred to kwargs_out
        for key in (keep+func_name_args):
            kwargs.pop(key)

    return kwargs_out

def format_str(string, kwargs, kwarg_aliases=None, kwarg_aliases_key='key_mapping'):
    """Format a string by substituting dictionary of values, also considering alternative names for the format keys
    according to a key aliases mapping.

    The kwarg_aliases_key allows one dict of formatting variables to be passed around for various purposes and
    include all the potential key mapping for different function signatures, without needing to pass it around as a
    separate parameter to many functions.

    Args:
        string: String to be formatted containing format fields eg "{pulse}_{camera}.nc"
        kwargs: Dict of values to substitute into format string
        kwarg_aliases: Dict of alternative key names that may occur in string, mapped to key names in kwargs
        kwarg_aliases_key: Key name in kwargs which if present should be treated as a source of kwarg_aliases

    Returns: String with format fields replaced by values in kwargs

    """
    try:
        string_out = string.format(**kwargs)
    except KeyError as e:
        if (kwarg_aliases is not None) or (kwarg_aliases_key in kwargs):
            if kwarg_aliases is None:
                kwarg_aliases = kwargs[kwarg_aliases_key]
            kwargs_extended = copy(kwargs)
            kwargs_alternative = {}
            for key, aliases in kwarg_aliases.items():
                for alias in make_iterable(aliases):
                    kwargs_alternative[alias] = kwargs[key]
            kwargs_extended.update(kwargs_alternative)
            string_out = string.format(**kwargs_extended)
        else:
            raise e
    return string_out

def increment_figlabel(label, i=2, suffix=' ({i})', start_With_siffix=False):
    num = label if not start_With_siffix else label + suffix.format(i=i)
    while num in plt.get_figlabels():
        num = label + suffix.format(i=i)
        i += 1
    return num


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
        # TODO: Move to utils?
        # TODO: fix latex display of axis labels
        # TODO: use this func in calcam_calibs get_surface_coords
    else:
        raise ValueError(f'Unexpected image data type {data}')
    return dataset

def delete_file(fn, path=None, ignore_exceptions=(), raise_on_fail=True, verbose=True):
    """Delete file with error handelling
    :param fn: filename
    :param path: optional path to prepend to filename
    :ignore_exceptions: Tuple of exceptions to pass over (but log) if raised eg (FileNotFoundError,)
    :param raise_on_fail: Raise exception if fail to delete file
    :param verbose   : Print log messages
    """
    fn = str(fn)
    if path is not None:
        path_fn = os.path.join(path, fn)
    else:
        path_fn = fn
    path_fn = os.path.abspath(os.path.expanduser(path_fn))
    success = False
    try:
        os.remove(path_fn)
        success = True
        if verbose:
            logger.info('Deleted file: {}'.format(path_fn))
    except ignore_exceptions as e:
        logger.debug(e)
    except Exception as e:
        if raise_on_fail:
            raise e
        else:
            logger.warning('Failed to delete file: {}'.format(path_fn))
    return success

def rm_files(paths, pattern, verbose=True, match=True, ignore_exceptions=()):
    """Delete files in paths matching patterns

    :param paths     : Paths in which to delete files
    :param pattern   : Regex pattern for files to delete
    :param verbose   : Print log messages
    :param match     : Use re.match instead of re.search (ie requries full pattern match)
    :param ignore_exceptions: Don't raise exceptions

    :return: None
    """
    paths = make_iterable(paths)
    for path in paths:
        path = str(Path(path).expanduser().resolve())
        if verbose:
            logger.info('Deleting files with pattern "{}" in path: {}'.format(pattern, path))
        for fn in os.listdir(path):
            if match:
                m = re.match(pattern, fn)
            else:
                m = re.search(pattern, fn)
            if m:
                delete_file(fn, path, ignore_exceptions=ignore_exceptions)
                if verbose:
                    logger.info('Deleted file: {}'.format(fn))
            else:
                pass

def logger_info(logger_arg):
    info = (f'Logger in "{logger_arg.name}" has:\n'
          f'parent "{logger_arg.parent}", \n'
          f'level "{logging.getLevelName(logger_arg.level)}", \n'
          f'effective level "{logging.getLevelName(logger_arg.getEffectiveLevel())}", \n'
          f'propagate = "{logger_arg.propagate}", \n'
          f'handlers {logger_arg.handlers}')
    return info


if __name__ == '__main__':
    pass
