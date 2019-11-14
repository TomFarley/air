#!/usr/bin/env python

"""Template for new movie pluglin for use with FIRE IR analysis code

1) Add code for your movie format to the two functions below.
2) Update "movie_plugin_name" variable (this must be unique)
3) Update desciription and add any other informations fields to the "plugin_info" variable
4) Update your .fire_config.json file to include this file in the "paths_input/movie_plugins"
5) Update your .fire_config.json file to include the name of your plugin under the relevent camera description
    "machines"/<my_machine>/"cameras"/<my_camera>/"movie_plugins"/[<movie_plugin_name>, ...]
"""

import logging
from typing import Union, Iterable, Tuple, Optional, Dict, Any
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

movie_plugin_name = '<my_plugin>'
plugin_info = {'description': 'This plugin reads <my format> format movie files'}

def read_movie_meta(pulse: int, camera: str, n_start:Optional[int]=None, n_end:Optional[int]=None,
                    stride:Optional[int]=1) -> Dict[str, Any]:
    """Return UDA movie object for given pulse, camera and frame range

    :param pulse: MAST-U pulse number
    :type pulse: int
    :param camera: MAST-U camera 3-letter diagnostic code (e.g. rir, rbb etc.)
    :type camera: str
    :param n_start: Frame number of first frame to load
    :type n_start: int
    :param n_end: Frame number of last frame to load
    :type n_end: int
    :param stride: interval between frames to be loaded
    :return: Movie meta data
    """
    raise NotImplementedError
    return movie_meta

def read_movie_data(pulse: int, camera: str, n_start:Optional[int]=None, n_end:Optional[int]=None,
                    stride:Optional[int]=1, transforms: Iterable[str]=None) \
                    -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return UDA movie object for given pulse, camera and frame range

    :param pulse: MAST-U pulse number
    :type pulse: int
    :param camera: MAST-U camera 3-letter diagnostic code (e.g. rir, rbb etc.)
    :type camera: str
    :param n_start: Frame number of first frame to load
    :type n_start: int
    :param n_end: Frame number of last frame to load
    :type n_end: int
    :param stride: interval between frames to be loaded
    :type stride: int
    :param transforms: List of of strings describing transformations to apply to frame data. Options are:
                    'reverse_x', 'reverse_y', 'transpose'
    :type transforms: Optional[Iterable[str]]
    :return: UDA movie object?
    """
    raise NotImplementedError
    return frame_nos, frame_times, frame_data

if __name__ == '__main__':
    pass