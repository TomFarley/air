#!/usr/bin/env python

"""Miscelanious utility functions

Created: 11-10-19
"""

import logging
from typing import Union, Iterable, Tuple, List, Optional, Any, Sequence
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

PathList = Sequence[Union[Path, str]]

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
    return pulse, camera.lower(), machine.lower()

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

def locate_file(paths: Iterable[Union[str, Path]], fns: Iterable[str],
                path_kws: Optional[dict]=None, fn_kws: Optional[dict]=None,
                return_raw_path: bool=False, return_raw_fn: bool=False,
                _raise: bool=True, verbose: Union[bool, int]=False) \
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
        _raise              : Raise an exception if the file is not located
        verbose             : Log whether or not the file was located

    Returns: (path, filename) / (None, None)

    """
    # TODO: detect multiple occurences/possible paths
    paths = make_iterable(paths)
    fns = make_iterable(fns)
    if path_kws is None:
        path_kws = {}
    if fn_kws is None:
        fn_kws = {}

    located = False
    for path_raw in paths:
        # Insert missing info in
        path_raw = str(path_raw)
        path = path_raw.format(**path_kws)
        try:
            path = Path(path).expanduser()
        except RuntimeError as e:
            if "Can't determine home directory" in str(e):
                continue
            else:
                raise e
        if not path.is_dir():
            continue
        path = path.resolve()
        for fn_raw in fns:
            fn = str(fn_raw).format(**fn_kws)
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
        message = f'Failed to locate file in paths "{paths}" with formats: {fns}, fn_kws: {fn_kws}'
        if _raise:
            raise IOError(message)
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
                               return_raw_fn=return_raw_fn, _raise=raise_, verbose=verbose)
            if f is not None:
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


