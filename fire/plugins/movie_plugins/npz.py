# -*- coding: future_fstrings -*-
#!/usr/bin/env python

"""Plugin functions for FIRE to read movie data from compressend numpy ".npz" files.

Two functions and two module level variables are required for this file to  function as a FIRE movie plugin:
    read_movie_data(pulse, camera, n_start, n_end, stride) -> (frame_nos, frame_times, frame_data)
    read_movie_meta(pulse, camera, n_start, n_end, stride) -> dict with minimum subset of keys:
        {'movie_format', 'n_frames', 'frame_range', 't_range', 'image_shape', 'fps', 'lens', 'exposure', 'bit_depth'}
    movie_plugin_name: str, typically same as module name
    plugin_info: dict, with description and any other useful information or mappings

Author: T. Farley
"""

import logging
from typing import Union, Iterable, Tuple, Optional, Any
from pathlib import Path
from copy import copy

import numpy as np
import pandas as pd

from fire.plugins.movie_plugins.ipx import get_detector_window_from_ipx_header

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

movie_plugin_name = 'npz'
plugin_info = {'description': 'This plugin reads movie data from npz numpy data files',
               'arg_name_mapping': {'camera': 'camera'}}

UDA_IPX_HEADER_FIELDS = ('board_temp', 'camera', 'ccd_temp', 'datetime', 'depth', 'exposure', 'filter', 'frame_times',
                         'gain', 'hbin', 'height', 'is_color', 'left', 'lens', 'n_frames', 'offset', 'preexp', 'shot',
                         'taps', 'top', 'vbin', 'view', 'width')

# Alternative variable names that can be used in npz movie files - different authors have different conventions
npz_key_map = {'time': 'frame_times', 'frame_nos': 'frame_numbers'}


def make_iterable(obj: Any, ndarray: bool = False,
                  cast_to: Optional[type] = None,
                  cast_dict: Optional = None,
                  # cast_dict: Optional[dict[type,type]]=None,
                  nest_types: Optional = None) -> Iterable:
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

def get_npz_movie_dict(path_fn: Union[str, Path]) -> dict:
    """Return dict of contents from npz file
    
    :param pulse: MAST-U pulse number
    :type pulse: int
    :param camera: MAST-U camera 3-letter diagnostic code (e.g. rir, rbb etc.)
    :type camera: str
    :param n_start: Frame number of first frame to load
    :type n_start: int
    :param n_end: Frame number of last frame to load
    :type n_end: int
    :param stride: interval between frames to be loaded
    :return: UDA movie object?
    """
    file = np.load(path_fn)
    file_dict = {key: file[key] for key in file.files}
    for key, value in file_dict.items():
        if isinstance(value, np.ndarray) and value.ndim == 0:
            file_dict[key] = value.item()
    file_dict = rename_dict_keys(file_dict, key_map=None)
    return file_dict

def read_movie_meta(path_fn: Union[str, Path], transforms: Iterable[str]=()) -> dict:
    """Read frame data from MAST IPX movie file format.

    :param path_fn: Path to IPX movie file
    :type path_fn: str, Path
    :param transforms: List of of strings describing transformations to apply to frame data. Options are:
                        'reverse_x', 'reverse_y', 'transpose'
    :type transforms: list
    :return: Dictionary of ipx file information
    :type: dict
    """
    # Read file header and first frame
    file_dict = get_npz_movie_dict(path_fn)
    frames = file_dict['frames']

    # Collect summary of ipx file meta data
    # file_header['ipx_version'] = vid.ipx_type
    movie_meta = {'movie_format': '.npz'}

    movie_meta['n_frames'] = len(frames)
    image_shape = np.array(frames.shape[1:])
    movie_meta['image_shape'] = image_shape

    if 'frame_times' in file_dict:
        t = file_dict['frame_times']
        movie_meta['t_range'] = np.array([np.min(t), np.max(t)])
        movie_meta['fps'] = 1/np.mean(np.diff(t))

    if 'frame_range' not in file_dict:
        logger.warning(f'Using temporary stand in value for meta data: "frame_range", "lens", "exposure"')
        movie_meta['frame_range'] = [0, len(frames)-1]

    if 'detector_window' not in file_dict:
        movie_meta['width'] = image_shape[1]
        movie_meta['height'] = image_shape[0]
        movie_meta['detector_window'] = get_detector_window_from_ipx_header(movie_meta, plugin='npz', fn=path_fn)

    meta_keys = ['n_frames', 'frame_range', 't_range', 'image_shape', 'fps', 'exposure', 'bit_depth', 'lens',
                 'detector_window']
    missing_keys = []
    missing_value = 'Unknown'
    for key in meta_keys:
        if key not in movie_meta:
            movie_meta[key] = file_dict.get(key, missing_value)
            missing_keys.append(key)
    if len(missing_keys) > 0:
        logger.warning(f'Missing meta data fields in npz movie set to "{missing_value}": {missing_keys}')

    file_dict.pop('frames')
    movie_meta['npz_header'] = file_dict

    return movie_meta

def rename_dict_keys(dict_original: dict, key_map: Optional[dict]=None) -> dict:
    """Rename dictionary keys

    :param dict_original: npz header dict with each parameter a separate scalar value
    :param key_map: mapping from old to new data key names
    :return: Reformatted dict
    """
    # raise NotImplementedError
    dict_new = copy(dict_original)
    if key_map is None:
        key_map = npz_key_map
    for old_key, new_key in key_map.items():
        try:
            dict_new[new_key] = dict_new.pop(old_key)
            logger.debug(f'Rename npz header parameter from "{old_key}" to "{new_key}".')
        except KeyError as e:
            logger.warning(f'Could not rename npz header parameter to "{new_key}" as parameter "{old_key}" not found.')

    return dict_new

def read_movie_data(path_fn: Union[str, Path],
                    n_start:Optional[int]=None, n_end:Optional[int]=None, stride:Optional[int]=1,
                    frame_numbers: Optional[Union[Iterable, int]]=None,
                    transforms: Optional[Iterable[str]]=()) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read frame data from MAST IPX movie file format.

    :param path_fn: Path to IPX movie file
    :type path_fn: str, Path
    :param frame_numbers: Frame numbers to read (should be monotonically increasing)
    :type frame_numbers: Iterable[int]
    :param transforms: List of of strings describing transformations to apply to frame data. Options are:
                        'reverse_x', 'reverse_y', 'transpose'
    :type transforms: Optional[Iterable[str]]
    :return: frame_nos, times, data_frames,
    :type: (np.array, np.array ,np.ndarray)
    """
    # Read file header and first frame
    file_dict = get_npz_movie_dict(path_fn)
    if 'frame_numbers' in file_dict:
        frame_numbers_all = file_dict['frame_numbers']
    else:
        frame_numbers_all = np.arange(file_dict['frames'].shape[0], dtype=int)
        logger.debug(f'Missing frame number info substituted with integers starting at 0')

    n_frames_movie = len(frame_numbers_all)
    if frame_numbers is None:
        if (n_start is not None) and (n_end is not None):
            frame_numbers = np.arange(n_start, n_end + 1, stride, dtype=int)

    if (transforms is not None) and (len(transforms) > 0):
        raise NotImplementedError

    frame_times = file_dict['frame_times']
    frame_data = file_dict['frames']

    if frame_numbers is not None:
        mask = is_in(frame_numbers_all, frame_numbers)
        return frame_numbers, frame_times[mask], frame_data[mask, :, :]
    else:
        return frame_numbers_all, frame_times, frame_data


if __name__ == '__main__':
    pulse = 30378
    camera = 'rir'
    # camera = 'air'
    n_start, n_end = 100, 110
    vid = get_uda_movie_obj(pulse, camera, n_start=n_start, n_end=n_end)
    # import pdb; pdb.set_trace()
    meta_data = read_movie_meta(pulse, camera, n_start, n_end)
    frame_nos, frame_times, frame_data = read_movie_data(pulse, camera, n_start, n_end)

    from fire.interfaces.uda_utils import import_pyuda
    pyuda, client = import_pyuda()
    r = client.list(pyuda.ListType.SIGNALS, shot=pulse, alias='air')
    signals = {}
    for d in r:
        signals[d.signal_name] = d.description
    print(signals)
    pass