# -*- coding: future_fstrings -*-
#!/usr/bin/env python

"""Miscelanious utility functions

Created: 11-10-19
"""

import logging, inspect, os, re, time
from typing import Union, Iterable, Tuple, List, Optional, Any, Sequence, Callable
from pathlib import Path
from copy import copy, deepcopy
from functools import partial

import numpy as np
import xarray as xr
from scipy import interpolate, stats
import pandas as pd
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

PathList = Sequence[Union[Path, str]]

try:
    string_types = (basestring, unicode)  # python2
except Exception as e:
    string_types = (str,)  # python3


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
                  nest_types: Optional=None,
                  ignore_types: Optional=()) -> Iterable:
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
        ignore_types: Types to not nest (eg if don't want to nest None)

    Returns:

    """
    if not isinstance(ignore_types, (tuple, list)):
        ignore_types = make_iterable(ignore_types, ndarray=False, ignore_types=())
    if (obj in ignore_types) or (type(obj) in ignore_types):
        # Don't nest this type of input
        return obj

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

def is_scalar(var, ndarray_0d=True, dataarray_0d=True):
    """ True if variable is scalar or string"""
    if isinstance(var, str):
        return True
    elif isinstance(var, (xr.DataArray)) and var.values.ndim == 0:
        return dataarray_0d
    elif hasattr(var, "__len__"):
        return False
    elif isinstance(var, (np.ndarray)) and var.ndim == 0:
        return ndarray_0d
    else:
        return True

def is_number(s, cast_string=False):
    """
    TODO: Test on numbers and strings and arrays
    """
    # from numbers import
    if (not cast_string) and isinstance(s, string_types):
        return False
    try:
        n=str(float(s))
        if n == "nan" or n=="inf" or n=="-inf" :
            return False
    except ValueError:
        try:
            complex(s)  # for complex
        except ValueError:
            return False
    except TypeError as e:  # eg trying to convert an array
        return False
    return True

def is_numeric(value):
    """Return True if value is a number or numeric array object, else False"""
    if isinstance(value, bool):
        numeric = False
    else:
        try:
            sum_values = np.sum(value)
            numeric = is_number(sum_values, cast_string=False)
        except TypeError as e:
            numeric = False
    return numeric

def str_to_number(string, cast=None, expect_numeric=False):
    """ Convert string to int if integer, else float. If cannot be converted to number just return original string
    :param string: string to convert number
    :param cast: type to cast output to eg always float
    :return: number
    """
    if isinstance(string, (int, float)):
        # leave unchanged
        return string
    if isinstance(string, str) and ('_' in string):
        # Do not convert strings with underscores and digits to numbers
        out = string
    else:
        try:
            out = int(string)
        except ValueError as e:
            try:
                out = float(string)
            except ValueError as e:
                out = string
    if isinstance(cast, type):
        out = cast(out)
    if not isinstance(out, (int, float)) and expect_numeric:
        raise ValueError('Input {string} could not be converted to a number'.format(string))
    return out

def ndarray_0d_to_scalar(array):
    """Convert 0D (single element) array to a scalar number (ie remove nested array)"""
    out = array
    if isinstance(array, (np.ndarray, xr.DataArray)) and array.ndim == 0:
        out = array.item()
    return out

def safe_len(var, scalar=1, all_nan=0, none=0, ndarray_0d=0, exclude_nans=False, **kwargs):
    """ Length of variable returning 1 instead of type error for scalars """
    # logger.debug(var)
    if var is None:
        return none
    elif isinstance(var, np.ndarray) and var.ndim == 0:
        return ndarray_0d
    elif is_scalar(var):  # checks if has atribute __len__ etc
        return scalar
    elif kwargs and var.__class__.__name__ in kwargs:
        return kwargs[var.__class__.__name__]
    else:
        assert hasattr(var, '__len__')
        try:
            if (len(np.array(var)) == np.sum(np.isnan(np.array(var)))):
                # If value is [Nan, Nan, ...] return zero length
                return all_nan
        except TypeError as e:
            pass
        if exclude_nans:
            var = np.array(var)
            nan_mask = np.isnan(var)
            return len(var[~nan_mask])
        else:
            return len(var)

def safe_arange(start, stop, step):
    """Return array of elements between start and stop, each separated by step.

    Replacement for np.arange that DOES always include stop.
    Normally np.arange should not include stop, but due to floating point precision sometimes it does, so output is
    unpredictable"""
    n = np.abs((stop - start) / step)
    if np.isclose(n, np.round(n)):
        # If n only differs from an integer by floating point precision, round it
        n = int(np.round(n))+1
    else:
        # If n is not approximately an integer, floor it
        n = int(np.floor(n))+1
        stop = start + (n-1)*step
    out = np.linspace(start, stop, n)
    return out

def safe_isnan(value, false_for_non_numeric=True):
    """Return false rather than throwing an error if the input type is not numeric"""
    try:
        out = np.isnan(value)
    except TypeError as e:
        if false_for_non_numeric:
            out = False
        else:
            raise e
    return out

def is_in(items, collection):
    """Return boolean mask, True for each item in items that is present in collection"""
    items = make_iterable(items)
    collection = make_iterable(collection)
    out = pd.Series(items).isin(collection).values
    return out

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    from https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def interpolate_out_nans(array_with_nans, interp_kind='linear', boundary_fill_value='extrapolate', **kwargs):
    """Replace nans in 1d array with interpolated values. Boundary nans are extrapolated.

    Args:
        array_with_nans: 1D array (ideally monotonically, steadily changing) containing gaps in data filled by nans
        boundary_fill_value: fill_value for sp.interpolate.interp1d
        interp_kind: interpolation method 'kind' for sp.interpolate.interp1d
        kwargs: Args to pass to sp.interpolate.interp1d
    Returns: 1D array with nans replaced by interpolated/extrapolated values

    """
    nan_mask, nonzero0 = nan_helper(array_with_nans)

    array_out = copy(array_with_nans)
    i_nan = nonzero0(nan_mask)
    i_not_nan = nonzero0(~nan_mask)
    y_not_nan = array_with_nans[~nan_mask]

    f = interpolate.interp1d(i_not_nan, y_not_nan, kind=interp_kind, fill_value=boundary_fill_value, **kwargs)
    array_out[nan_mask] = f(i_nan)

    return array_out


def func_name(level=0):
    return inspect.stack()[level+1][3]

def module_name(level=0):
    """ Return name of the module level levels from where this
    function was called. level = 1 goes 1 level further up the stack """
    # return inspect.getmodulename(inspect.stack()[level+1][1])
    # print 'tf_debug.py, 85:', os.path.basename(inspect.stack()[level+1][1])
    try:
        name = os.path.basename(inspect.stack()[level+1][1])
    except IndexError:
        print('tf_debug: Failed to return module name. See stack:')
        try:
            print(inspect.stack())
        except:
            print("inspect module doesn't seem to be working at all!")
        name = '*UNKNOWN*'
    return name

def line_no(level=0):
    """ Return line number level levels from where this
    function was called. level = 1 goes 1 level further up the stack """
    try:
        line = str(inspect.stack()[level+1][2])
    except IndexError:
        print('tf_debug: Failed to return line number. See stack:')
        try:
            print(inspect.stack())
        except:
            print("inspect module doesn't seem to be working at all!")
        line = '*UNKNOWN*'
    return line

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
    """Convert eg "None" -> None, "true" -> True, "[1,2]" -> [1,2]

    Args:
        df:
        col_subset:
        allow_strings:
        list_delimiters:
        strip_chars:

    Returns:

    """
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

def add_aliases_to_dict(dict_in, aliases, remove_original=False):
    """Add alternative names for fields"""
    for original_key, new_keys in aliases.items():
        for new_key in make_iterable(new_keys):
            dict_in[new_key] = dict_in[original_key]
    if remove_original:
        for key in aliases:
            dict_in.pop(key)
    return dict_in

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
        for key in make_iterable(required):
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

def chunks(lst, n):
    """Yield successive n-sized chunks from lst.

    Args:
        lst: Series object eg list or string
        n: Chunk size

    Returns: Generator for chunks

    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def increment_figlabel(label, i=2, suffix=' ({i})', start_With_siffix=False):
    num = label if not start_With_siffix else label + suffix.format(i=i)
    while num in plt.get_figlabels():
        num = label + suffix.format(i=i)
        i += 1
    return num


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

def func_name(level=0):
    return inspect.stack()[level+1][3]

def module_name(level=0):
    """ Return name of the module level levels from where this
    function was called. level = 1 goes 1 level further up the stack """
    # return inspect.getmodulename(inspect.stack()[level+1][1])
    # print 'tf_debug.py, 85:', os.path.basename(inspect.stack()[level+1][1])
    try:
        name = os.path.basename(inspect.stack()[level+1][1])
    except IndexError:
        print('tf_debug: Failed to return module name. See stack:')
        try:
            print(inspect.stack())
        except:
            print("inspect module doesn't seem to be working at all!")
        name = '*UNKNOWN*'
    return name

def line_no(level=0):
    """ Return line number level levels from where this
    function was called. level = 1 goes 1 level further up the stack """
    try:
        line = str(inspect.stack()[level+1][2])
    except IndexError:
        print('tf_debug: Failed to return line number. See stack:')
        try:
            print(inspect.stack())
        except:
            print("inspect module doesn't seem to be working at all!")
        line = '*UNKNOWN*'
    return line

def get_traceback_location(level=0, format='{module_name}:{func_name}:{line_no}'):
    """Returns an informative prefix for verbose Debug output messages"""
    module = module_name(level=level)
    func = func_name(level=level)
    line = line_no(level=level)
    return format.format(module_name=module, func_name=func, line_no=line)

def whereami(level=0):
    """ Return a string detailing the line number, function name and filename from level relative to where this
    function was called """

    # string = module_name(level=level+1)+', '+func_name(level=level+1)+', '+line_no(level=level+1)+': '
    string = line_no(level=level+1)+', '+func_name(level=level+1)+', '+module_name(level=level+1)+':\t'
    return string

def file_line(level=1):
    """ Return string containing filename and line number at level """
    return module_name(level=level+1)+', '+line_no(level=level+1)+': '


def traceback(level=0):
    """ Return string listing the full fraceback at the level relative to where this function was called """
    string = 'Traceback:\n'
    while not (func_name(level=level) == '<module>'):
        string += line_no(level=level+1)+', '+func_name(level=level+1)+', '+module_name(level=level+1)+'\n'
        level += 1
    return string.rstrip()

def print_progress(iteration, total, prefix='', suffix='', frac=False, t0=None,
                  decimals=2, nth_loop=2, barLength=50, flush=True):
    """
    Based on http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console

    Call at start of a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration starting at 0 (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    # TODO: convert to class with __call__ (print 0% on __init__) - add to timeline class
    # TODO: Change/add nth_loop to min time between updates
    # TODO: Add compatibility for logger handlers
    # TODO: Make bar optional
    if (iteration % nth_loop != 0) and (
            iteration != total - 1):  # Only print every nth loop to reduce slowdown from printing
        return
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
    filledLength = int(round(barLength * iteration / float(total)))
    percents = round(100.00 * (iteration / float(total)), decimals)
    bar = '|' * filledLength + '-' * (barLength - filledLength)
    frac = '{}/{} '.format(iteration, total) if frac else ''
    if t0 is None:
        time = ''
    else:
        if isinstance(t0, float):
            # Convert float time from time.time() (seconds since the Epoch) to datetime
            t0 = datetime.fromtimestamp(t0)
        t1 = datetime.now()
        t_diff_past = relativedelta(t1, t0)  # time past in loop
        mul = float(total - iteration) / iteration if iteration > 0 else 0
        t_diff_rem = t_diff_past * mul  # estimate of remaining time
        t_diff_past = '({h}h {m}m {s}s)'.format(h=t_diff_past.hours, m=t_diff_past.minutes, s=t_diff_past.seconds)
        if t_diff_rem.hours > 0:  # If expected to take over an hour display date and time of completion
            t_diff_rem = (datetime.now() + t_diff_rem).strftime("(%d/%m/%y %H:%M)")
        else:  # Display expected time remaining
            t_diff_rem = '({h}h {m}m {s}s)'.format(h=t_diff_rem.hours, m=t_diff_rem.minutes, s=t_diff_rem.seconds)
        if mul == 0:
            t_diff_rem = '?h ?m ?s'
        time = ' {past} -> {remain}'.format(past=t_diff_past, remain=t_diff_rem)

    sys.stdout.write('\r %s |%s| %s%s%s%s %s' % (prefix, bar, frac, percents, '%', time, suffix)),
    if flush:
        sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()


def compare_dict(dict1, dict2, tol=1e-12, top=True):
    """ Recursively check that two dictionaries and all their constituent sub dictionaries have the same numerical
    values. Avoids floating point precision comparison problems.
    """
    assert(isinstance(dict1, dict) and isinstance(dict2, dict))
    from collections import Counter
    if Counter(dict1.keys()) != Counter(dict2.keys()):  # Use counter to ignore order (if objects are hashable)
        keys_unique_1 = [k for k in dict1 if k not in dict2]
        keys_unique_2 = [k for k in dict2 if k not in dict1]
        keys_common = [k for k in dict2 if k in dict1]
        print(f'def compare_numeric_dict: Dictionaries have different keys:\n'
              f'keys_common: {keys_common}\ndict1_unique: {keys_unique_1}\ndict2_unique: {keys_unique_2}')
        return False

    for key in dict1.keys():
        if isinstance(dict1[key], dict) or isinstance(dict2[key], dict):

            if not (isinstance(dict1[key], dict) and isinstance(dict2[key], dict)):
                logger.debug('Dictionaries are different - One value is a dict while the other is not')
                return False
            if compare_dict(dict1[key], dict2[key], top=False) is False:
                return False
        # elif isinstance(dict2[key], dict):
        #     if compare_numeric_dict(dict1, dict2[key], top=False) is False:
        #         return False
        else:
            try:
                if np.abs(dict1[key]-dict2[key]) > tol:  # numerical
                    return False
            except TypeError:
                if dict1[key] != dict2[key]:  # string etc
                    return False
    return True


def is_subset(subset, full_set):
    """Return True if all elements of subset are in fullset"""
    return set(subset).issubset(set(full_set))

def args_for(func, kwargs, include=(), exclude=(), match_signature=True, named_dict=True, remove=True):
    """Return filtered dict of args from kwargs that match input for func.
    :param - Effectively filters kwargs to return those arguments
    :param - func            - function(s) to provide compatible arguments for
    :param - kwargs          - list of kwargs to filter for supplied function
    :param - exclude         - list of kwargs to exclude from filtering
    :param - match_signature - apply filtering to kwargs based on func call signature
    :param - named_dict      - if kwargs contains a dict under key '<func_name>_args' return its contents (+ filtered kwargs)
    :param - remove          - remove filtered kwargs from original kwargs
    """
    #TODO: Include positional arguments!
    func = make_iterable(func)  # Nest lone function in list for itteration, TODO: Handle itterable classes
    kws = {}
    keep = []  # list of argument names
    name_args = []
    for f in func:
        # Add arguments for each function to list of arguments to keep
        if isinstance(f, type):
            # If a class look at it's __init__ method
            keep += list(inspect.signature(f.__init__).parameters.keys())
        else:
            keep += list(inspect.signature(f).parameters.keys())
        name_args += ['{name}_args'.format(name=f.__name__)]
    if match_signature:
        matches = {k: v for k, v in kwargs.items() if (((k in keep) and (k not in exclude)) or (k in include))}
        kws.update(matches)
    if named_dict:  # Look for arguments <function>_args={dict of keyword arguments}
        keep_names = {k: v for k, v in kwargs.items() if (k in name_args)}
        kws.update(keep_names)
    if remove:  # Remove key value pairs from kwargs that were transferred to kws
        for key in kws:
            kwargs.pop(key)
    return kws


def argsort(itterable):
    #http://stackoverflow.com/questions/3382352/equivalent-of-numpy-argsort-in-basic-python/3382369#3382369
    #by unutbu
    try:
        out = sorted(range(len(itterable)), key=itterable.__getitem__)
    except TypeError as e:
        itterable = [str(val) for val in itterable]
        out = sorted(range(len(itterable)), key=itterable.__getitem__)
    return out

def args_for(func, kwargs, include=(), exclude=(), match_signature=True, named_dict=True, remove=True):
    """Return filtered dict of args from kwargs that match input for func.
    :param - Effectively filters kwargs to return those arguments
    :param - func            - function(s) to provide compatible arguments for
    :param - kwargs          - list of kwargs to filter for supplied function
    :param - exclude         - list of kwargs to exclude from filtering
    :param - match_signature - apply filtering to kwargs based on func call signature
    :param - named_dict      - if kwargs contains a dict under key '<func_name>_args' return its contents (+ filtered kwargs)
    :param - remove          - remove filtered kwargs from original kwargs
    """
    #TODO: Include positional arguments!
    func = make_iterable(func)  # Nest lone function in list for itteration, TODO: Handle itterable classes
    kws = {}
    keep = []  # list of argument names
    name_args = []
    for f in func:
        # Add arguments for each function to list of arguments to keep
        if isinstance(f, type):
            # If a class look at it's __init__ method
            keep += list(inspect.signature(f.__init__).parameters.keys())
        else:
            keep += list(inspect.signature(f).parameters.keys())
        name_args += ['{name}_args'.format(name=f.__name__)]
    if match_signature:
        matches = {k: v for k, v in kwargs.items() if (((k in keep) and (k not in exclude)) or (k in include))}
        kws.update(matches)
    if named_dict:  # Look for arguments <function>_args={dict of keyword arguments}
        keep_names = {k: v for k, v in kwargs.items() if (k in name_args)}
        kws.update(keep_names)
    if remove:  # Remove key value pairs from kwargs that were transferred to kws
        for key in kws:
            kwargs.pop(key)
    return kws

def mode_simple(data):
    data = make_iterable(data, ndarray=True)
    if len(data) > 0:
        mode = stats.mode(data).mode[0]
    else:
        mode = np.nan
    # mode = ndarray_0d_to_scalar(mode)
    return mode

def format_str_partial(string, format_kwargs, allow_partial=True):
    """Format a string allowing partial substitution of keys

    Args:
        string: Input format string or existing partially formatting string object
        format_kwargs: Dict of values to substitute into string
        allow_partial: If false raise KeyErrors as for normal string formatting

    Returns:

    """
    if isinstance(string, str):
        try:
            string = string.format(**format_kwargs)
        except KeyError as e:
            if not allow_partial:
                raise e
            string = partial(string.format, **format_kwargs)
    elif isinstance(string, partial):
        try:
            string = string(**format_kwargs)
        except KeyError as e:
            string = partial(string, **format_kwargs)
    # TODO: Find kws in string.func.__self__ not persent in string.keywords and sub with value
    return string

def in_freia_batch_mode():
    """Return True if current python interpreter is being run as a batch job (ie no display for plotting etc)"""
    batch_mode = os.getenv('LOADL_ACTIVE', None)
    return batch_mode == 'yes'

def ask_input_yes_no(message, suffix=' ([Y]/n)? ', message_format='{message}{suffix}', default_yes=True,
                     batch_mode_default=True, sleep=0.1):
    """Ask yes/no question to raw input"""
    if in_freia_batch_mode():
        return batch_mode_default
    if default_yes is False:
        suffix = ' (y/[N])? '
    if sleep:
        # Make sure logging output has time to clear before prompt is printed
        time.sleep(sleep)
    question = message_format.format(message=message, suffix=suffix)
    answer = input(question)
    accept = ['y', 'yes']
    if default_yes:
        accept.append('')
    if answer.lower() in accept:
        out = True
    else:
        out = False
    return out

def filter_non_builtins(obj, parent=None, ind=None, additional_types=(np.ndarray, xr.DataArray, xr.Dataset, Path),
                        copy_=False):
    import builtins
    builtin_types = tuple((t for t in builtins.__dict__.values()))

    if copy_:
        try:
            obj = deepcopy(obj)
        except AttributeError as e:
            try:
                obj = copy(obj)
            except AttributeError as e:
                logger.warning('Failed to deepcopy obj before filtering non-builtins')

    accepted_types = builtin_types + additional_types
    if safe_len(obj, scalar=0) > 0:
        if isinstance(obj, dict):
            for key, item in obj.items():
                obj, parent = filter_non_builtins(item, parent=obj, ind=key)
        else:
            for i, item in enumerate(obj):
                obj, parent = filter_non_builtins(item, parent=obj, ind=i)
    else:
        if type(obj) not in accepted_types:
            logger.info(f'{type(obj)} not in accepted types. Obj: {obj}')
            if isinstance(parent, (dict, list)):
                parent.pop(ind)
            if isinstance(parent, (tuple, set)):
                parent = type(parent)((item for item in parent if item is not obj))
            elif parent is None:
                raise TypeError('Whole object is non-builtin')
            else:
                raise NotImplementedError(f'Unexpected type {type(obj)} not in accepted types. Obj: {obj}')
    return obj, parent


def filter_nested_dict_key_paths(dict_in, key_paths_keep, compress_key_paths=True, path_fn=None):
    """Return a subset of nested dicts for given paths

    Args:
        dict_in: dict of dicts
        key_paths_keep: iterable of key names navigating nested dict structure
        compress_key_paths: Only keep last key in key_path eg key_path=('mast', 29852, 'signal') -> {'signal': 'rir'}
        path_fn:            Name/path of jason file being filtered (only used for error messages)

    Returns: Filtered nested dict

    """

    if key_paths_keep is not None:
        key_paths_keep = make_iterable(key_paths_keep)
        dict_out = {}
        for key_path in key_paths_keep:
            key_path = make_iterable(key_path)
            subset = dict_in
            for key in key_path:
                try:
                    subset = subset[key]
                except KeyError as e:
                    raise KeyError(f'json file ({path_fn}) does not contain key "{key}" in key path "{key_path}":\n'
                                   f'{subset}')
            if compress_key_paths:
                # Compress the key path into just the last key
                dict_out[key_path[-1]] = subset
            else:
                for i, key in enumerate(key_path):
                    if i == len(key_path)-1:
                        dict_out[key] = subset
                    else:
                        if key not in dict_out:
                            dict_out[key] = {}
    else:
        dict_out = dict_in

    return dict_out


def drop_nested_dict_key_paths(dict_in, key_paths_drop):
    """Return a subset of nested dicts for given paths

    Args:
        dict_in: dict of dicts
        key_paths_drop: iterable of key names navigating nested dict structure to drop

    Returns: Filtered nested dict

    """
    dict_out = copy(dict_in)
    if (key_paths_drop is not None):
        for key_path in make_iterable(key_paths_drop):
            key_path = make_iterable(key_path)
            subset = dict_out
            for i, key in enumerate(key_path):
                if i == len(key_path)-1:
                    if key in subset:
                        subset.pop(key)
                else:
                    try:
                        subset = subset[key]
                    except KeyError as e:
                        raise KeyError(f'json file ({path_fn}) does not contain key "{key}" in key path "{key_path}":\n'
                                       f'{subset}')
    return dict_out


def cast_lists_in_dict_to_arrays(dict_in, raise_on_no_lists=False):
    dict_out = deepcopy(dict_in)
    n_lists_converted = 0

    for key, value in dict_out.items():
        if isinstance(value, (list)):
            dict_out[key] = cast_nested_lists_to_arrays(value, raise_on_no_lists=False)
            n_lists_converted += 1
        elif isinstance(value, dict):
            dict_out[key] = cast_lists_in_dict_to_arrays(value, raise_on_no_lists=False)
        else:
            pass

    if raise_on_no_lists and (n_lists_converted==0):
        raise TypeError(f'Input dict does not contain any lists to convert to ndarrays: {dict_in}')

    return dict_out


def cast_nested_lists_to_arrays(list_in, raise_on_no_lists=False):
    list_out = deepcopy(list_in)
    n_lists_converted = 0

    if all_list_elements_same_type(list_in, invalid_types=(list, tuple)):
        list_out = np.array(list_in)
        n_lists_converted += 1
    else:
        for i, value in enumerate(list_in):
            if isinstance(value, (list)):
                if all_list_elements_same_type(value, invalid_types=(list, tuple)):
                    list_out[i] = np.array(value)
                else:
                    list_out[i] = cast_nested_lists_to_arrays(value, raise_on_no_lists=False)
                if isinstance(list_out[i], np.ndarray):
                    n_lists_converted += 1
            elif isinstance(value, dict):
                list_out[i] = cast_lists_in_dict_to_arrays(value, raise_on_no_lists=False)
            else:
                pass

    if raise_on_no_lists and (n_lists_converted == 0):
        raise TypeError(f'Input list does not contain any lists to convert to ndarrays: {list_in}')

    return list_out


def cast_lists_to_arrays(list_or_dict):
    if isinstance(list_or_dict, list):
        out = cast_nested_lists_to_arrays(list_or_dict)
    elif isinstance(list_or_dict, dict):
        out = cast_lists_in_dict_to_arrays(list_or_dict)
    else:
        raise TypeError(f'Input is not list or dict: {list_or_dict}')

    return out


def all_list_elements_same_type(list_in, invalid_types=(list, tuple)):
    types = [type(item) for item in list_in]
    ref_type = types[-1]
    all_types_same = all(t == ref_type for t in types)
    if all_types_same and (ref_type in invalid_types):
        all_types_same = False
    return all_types_same


def two_level_dict_to_multiindex_df(d):
    """Convert nested dictionary to two level multiindex dataframe

    Args:
        d: Input nested dictionary

    Returns: DataFrame containing contents of d

    Examples:
        d:
        {"MAST_S1_L3_centre_radial_1":
            {
              "start": {"R": 0.769, "z": -1.827, "phi": 70.6},
              "end": {"R": 1.484, "z": -1.831, "phi": 70.6}
            }
        }
        df:
                                                  R      z   phi
            MAST_S1_L3_centre_radial_1 end    1.484 -1.831  70.6
                                       start  0.769 -1.827  70.6

    """
    df = pd.DataFrame.from_dict({(k1, k2): v2 for k1, v1 in d.items() for k2, v2 in v1.items()}, orient='index')
    return df