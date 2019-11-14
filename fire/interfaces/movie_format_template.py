#!/usr/bin/env python

"""


Created: 
"""

import logging
from typing import Union, Iterable, Tuple, Optional
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def read_movie_meta(pulse: int, camera: str, n_start:Optional[int]=None, n_end:Optional[int]=None,
                    stride:Optional[int]=1):
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
                    stride:Optional[int]=1, transforms: Iterable[str]=None):
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