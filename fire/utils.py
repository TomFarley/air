# -*- coding: future_fstrings -*-
#!/usr/bin/env python

"""Miscelanious utility functions

Created: 11-10-19
"""

import logging, inspect
from typing import Union, Iterable, Tuple, List, Optional, Any, Sequence
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

PathList = Sequence[Union[Path, str]]

def movie_data_to_xarray(frame_data, frame_times, frame_nos=None):
    """Return frame data in xarray.DataArray object

    Args:
        frame_data  : Array of camera digit level data with dimensions [t, y, x]
        frame_times : Array of frame times
        frame_nos   : Array of frame numbers

    Returns: DataArray of movie data

    """
    if frame_nos is None:
        frame_nos = np.arange(frame_data.shape[0])
    frame_data = xr.DataArray(frame_data, dims=['t', 'y_pix', 'x_pix'],
                              coords={'t': frame_times, 'n': ('t', frame_nos),
                                      'y_pix': np.arange(frame_data.shape[1]),
                                      'x_pix': np.arange(frame_data.shape[2])},
                              name='frame_data')
    # Default to indexing by frame number
    frame_data = frame_data.swap_dims({'t': 'n'})
    frame_data.attrs.update({
        'long_name': 'DL',
        'units': 'arb',
        'description': 'Digit level (DL) intensity counts recorded by camera sensor, dependent on photon flux'})
    frame_data['n'].attrs.update({
        'long_name': '$n_{frame}$',
        'units': '',
        'description': 'Camera frame number (integer)'})
    frame_data['t'].attrs.update({
        'long_name': '$t$',
        'units': 's',
        'description': 'Camera frame time'})
    frame_data['x_pix'].attrs.update({
        'long_name': '$x_{pix}$',
        'units': '',
        'description': 'Camera x pixel coordinate'})
    frame_data['y_pix'].attrs.update({
        'long_name': '$y_{pix}$',
        'units': '',
        'description': 'Camera y pixel coordinate'})
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

if __name__ == '__main__':
    pass

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
    if isinstance(cast_to, type):
        if cast_to == np.ndarray:
            obj = np.array(obj)
        else:
            obj = cast_to(obj)  # cast to new type eg list
    return obj

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
            path = Path(path).expanduser().resolve()
        except RuntimeError as e:
            if "Can't determine home directory" in str(e):
                pass
                # continue
            else:
                raise e
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
    paths_exist, paths_raw_exist, paths_raw_not_exist = dirs_exist(paths, path_kws=path_kws)
    for path, path_raw in zip(paths_exist, paths_raw_exist):
        for fn_raw in fns:
            try:
                fn = str(fn_raw).format(**fn_kws)
            except KeyError as e:
                raise ValueError(f'Cannot locate file without value for "{e.args[0]}": "{fn_raw}", {fn_kws}"')
            except IndexError as e:
                raise e
            fn_path = path / fn
            if fn_path.is_file():
                located = True
                path_out = path_raw if return_raw_path else path
                fn_out = fn_raw if return_raw_fn else fn
                if verbose >= 2:
                    logging.info('Located "{}" in {}'.format(fn_out, path_out))
                break
        if located:
            break
    else:
        # File not located
        message = f'Failed to locate file with formats: {fns} in paths:\n"{paths}"\nwith fn_kws:\n{fn_kws}'
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

def filter_kwargs(func, kwargs, include=(), exclude=(), match_signature=True, named_dict=True, remove=True):
    """Return filtered dict of kwargs that match input call signature for supplied function.

    Args:
        func            : Function(s) to provide compatible arguments for
        kwargs          : List of kwargs to filter for supplied function
        include         : List of kwargs to include regardless of function signature
        exclude         : List of kwargs to exclude regardless of function signature
        match_signature : Whether to use use function signature to filter kwargs
        named_dict      : If kwargs contains a dict under key '<func_name>_args' return its contents (+ filtered kwargs)
        remove          : Remove filtered kwargs from original kwargs dict

    Returns: Dictionary of keyword arguments that are compatible with the supplied function's call signature

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

