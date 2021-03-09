# -*- coding: future_fstrings -*-
#!/usr/bin/env python

"""
This module defines functions for interfacing with MAST (2000-2013) data archives and systems.
"""

import logging
from typing import Union, Iterable, Tuple, Optional
from pathlib import Path
import numbers
from copy import copy

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from pyIpx.movieReader import imstackReader
from fire.interfaces.interfaces import json_load
from fire.plugins.movie_plugins.ipx import get_detector_window_from_ipx_header
from fire.interfaces.camera_data_formats import read_ircam_raw_int16_sequence_file

logging.basicConfig()
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

movie_plugin_name = 'raw_movie'
plugin_info = {'description': "This plugin reads raw movie files output by the IRCAM Works software, with meta data "
                              "supplied from a json file in the same directory"}


def read_movie_meta(path_fn: Union[str, Path], raise_on_missing_meta=True) -> dict:
    """Read frame data from imstack image directory ('png','jpg','bmp','tif')

    :param path_fn: Path to imstack image directory
    :type path_fn: str, Path
    :param transforms: List of of strings describing transformations to apply to frame data. Options are:
                        'reverse_x', 'reverse_y', 'transpose'
    :type transforms: list
    :return: Dictionary of ipx file information
    :type: dict
    """
    movie_meta = {}

    # If present, also read meta data from json file with movie file
    path_fn = Path(path_fn)
    path = path_fn.parent if path_fn.is_file() else path_fn
    path_fn_meta = path / 'movie_meta_data.json'
    if not path_fn_meta.exists() and path_fn.is_file():
        path_fn_meta = path / (str(path_fn.stem) + '_meta.json')

    movie_meta_json = json_load(path_fn_meta, raise_on_filenotfound=False, lists_to_arrays=True)

    if isinstance(movie_meta_json, list):
        movie_meta_json = dict(movie_meta_json)  # Saved as json list in order to save ints, floats etc

    if isinstance(movie_meta_json, dict):
        frame_times = movie_meta_json.get('frame_times', None)
        movie_meta.update(movie_meta_json)
        if (frame_times is not None) and ('t_range' not in movie_meta):
            movie_meta['t_range'] = np.array([np.min(frame_times), np.max(frame_times)])
    else:
        message = f'Raw movie does not have a meta data json file: {path}'
        if raise_on_missing_meta:
            raise IOError(message)
        else:
            logger.warning(message)
    return movie_meta

def read_movie_data(path_fn: Union[str, Path],
                    n_start:Optional[int]=None, n_end:Optional[int]=None, stride:Optional[int]=1,
                    frame_numbers: Optional[Union[Iterable, int]]=None,
                    transforms: Optional[Iterable[str]]=(), grayscale: bool=True) -> Tuple[np.ndarray, np.ndarray,
                                                                                     np.ndarray]:
    """Read frame data from imstack image directory ('png','jpg','bmp','tif').

    :param path_fn: Path to imstack directory
    :type path_fn: str, Path
    :param frame_numbers: Frame numbers to read (should be monotonically increasing)
    :type frame_numbers: Iterable[int]
    :param transforms: List of of strings describing transformations to apply to frame data. Options are:
                        'reverse_x', 'reverse_y', 'transpose'
    :type transforms: Optional[Iterable[str]]
    :return: frame_nos, times, data_frames,
    :type: (np.array, np.array ,np.ndarray)
    """
    if transforms is None:
        transforms = ()
    path_fn = Path(path_fn)
    if not path_fn.exists():
        raise FileNotFoundError(f'Raw file not found: {path_fn}')
    if frame_numbers is not None:
        raise NotImplementedError

    # raw_frame_stack_to_images
    frame_numbers, frame_data = read_ircam_raw_int16_sequence_file(path_fn, n_start=n_start, n_end=n_end)

    if frame_data.ndim == 2:
        frame_data = frame_data[np.newaxis, :, :]

    meta_data = read_movie_meta(path_fn)

    if 'frame_numbers' in meta_data:
        frame_numbers_all = meta_data['frame_numbers']
    # else:
    #     frame_numbers_all = np.arange(len(frame_data))

    if 'frame_times' in meta_data:
        frame_times = meta_data['frame_times']
        if (n_start is not None) or (n_end is not None):
            if n_start is None:
                n_start = frame_numbers[0]
            if n_end is None:
                n_end = frame_numbers[-1]
            i_frames = np.nonzero((frame_numbers_all >= n_start) & (frame_numbers_all <= n_end))
            # frame_numbers = frame_numbers[i_frames]
            frame_times = frame_times[i_frames]
            # frame_data = frame_data[i_frames]
    else:
        frame_times = np.full_like(frame_numbers, np.nan)

    if (transforms is not None) and (len(transforms) > 0):
        raise NotImplementedError

    return frame_numbers, frame_times, frame_data


if __name__ == '__main__':
    # directory = Path('/projects/SOL/Data/Cameras/SA1/29496/C001H001S0001/').resolve()
    directory = Path('/home/tfarley/data/movies/mast_u/50000/rir/').resolve()
    # frame_nos = np.arange(100, 202)
    frame_nos = None
    meta_data = read_movie_meta(directory)
    frame_nos, frame_times, frame_data = read_movie_data(directory, frame_numbers=frame_nos)
    print(meta_data)
    plt.imshow(frame_data[0])
    pass