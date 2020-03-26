# -*- coding: future_fstrings -*-
#!/usr/bin/env python

"""Plugin functions for FIRE to read movie data from compressend numpy ".npz" files.

Two functions and two module level variables are required for this file to  function as a FIRE movie plugin:
    read_movie_data(pulse, camera, n_start, n_end, stride) -> (frame_nos, frame_times, frame_data)
    read_movie_meta(pulse, camera, n_start, n_end, stride) -> dict with minimum subset of keys:
        {'movie_format', 'n_frames', 'frame_range', 't_range', 'frame_shape', 'fps', 'lens', 'exposure', 'bit_depth'}
    movie_plugin_name: str, typically same as module name
    plugin_info: dict, with description and any other useful information or mappings

Author: T. Farley
"""

import logging
from typing import Union, Iterable, Tuple, Optional
from pathlib import Path
from copy import copy

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

try:
    import pyuda
    client = pyuda.Client()
except ImportError as e:
    logger.warning(f'Failed to import pyuda. ')
    pyuda = False

movie_plugin_name = 'npz'
plugin_info = {'description': 'This plugin reads movie data from npz numpy data files',
               'arg_name_mapping': {'camera': 'camera'}}

UDA_IPX_HEADER_FIELDS = ('board_temp', 'camera', 'ccd_temp', 'datetime', 'depth', 'exposure', 'filter', 'frame_times',
                         'gain', 'hbin', 'height', 'is_color', 'left', 'lens', 'n_frames', 'offset', 'preexp', 'shot',
                         'taps', 'top', 'vbin', 'view', 'width')

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
    file_dict = {key: getattr(file, key) for key in file.files}
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

    # Collect summary of ipx file meta data
    # file_header['ipx_version'] = vid.ipx_type
    movie_meta = {'movie_format': '.ipx'}
    meta_keys = ['n_frames', 'frame_range', 't_range', 'frame_shape', 'fps', 'exposure', 'bit_depth', 'lens']
    for key in meta_keys:
        if key not in file_dict:
            logger.warning(f'Missing "{key}" meta data in npz movie. Setting to None.')
        movie_meta[key] = file_dict.get(key, None)

    return movie_meta

def rename_dict_keys(dict_original: dict, key_map: Optional[dict]=None) -> dict:
    """Rename dictionary keys

    :param dict_original: npz header dict with each parameter a separate scalar value
    :param key_map: mapping from old to new data key names
    :return: Reformatted dict
    """
    raise NotImplementedError
    dict_new = copy(dict_original)
    if key_map is None:
        key_map = {}
    for old_key, new_key in key_map.items():
        try:
            dict_new[new_key] = dict_new.pop(old_key)
            logger.debug(f'Rename npz header parameter from "{old_key}" to "{new_key}".')
        except KeyError as e:
            logger.warning(f'Could not rename npz header parameter to "{new_key}" as paremeter "{old_key}" not found.')

    return dict_new

def read_movie_data(path_fn: Union[str, Path], frame_nos: Optional[Union[Iterable, int]]=None,
                    transforms: Optional[Iterable[str]]=()) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read frame data from MAST IPX movie file format.

    :param path_fn: Path to IPX movie file
    :type path_fn: str, Path
    :param frame_nos: Frame numbers to read (should be monotonically increasing)
    :type frame_nos: Iterable[int]
    :param transforms: List of of strings describing transformations to apply to frame data. Options are:
                        'reverse_x', 'reverse_y', 'transpose'
    :type transforms: Optional[Iterable[str]]
    :return: frame_nos, times, data_frames,
    :type: (np.array, np.array ,np.ndarray)
    """
    # Read file header and first frame
    file_dict = get_npz_movie_dict(path_fn)
    frame_nos = file_dict['frame_numbers']
    frame_times = file_dict['time']
    frame_data = file_dict['frames']
    return frame_nos, frame_times, frame_data


if __name__ == '__main__':
    pulse = 30378
    camera = 'rir'
    # camera = 'air'
    n_start, n_end = 100, 110
    vid = get_uda_movie_obj(pulse, camera, n_start=n_start, n_end=n_end)
    # import pdb; pdb.set_trace()
    meta_data = read_movie_meta(pulse, camera, n_start, n_end)
    frame_nos, frame_times, frame_data = read_movie_data(pulse, camera, n_start, n_end)

    r = client.list(pyuda.ListType.SIGNALS, shot=pulse, alias='air')
    signals = {}
    for d in r:
        signals[d.signal_name] = d.description
    print(signals)
    pass