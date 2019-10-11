#!/usr/bin/env python

""" 
Author: T. Farley
"""

import logging
from typing import Dict, Iterable, Optional

import numpy as np
import xarray as xr

import pyuda

# fileConfig('../logging_config.ini')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

client = pyuda.Client()

def return_true():
    return True

def get_frames_uda(shot: int, camera: str, n_start:Optional[int]=None, n_end:Optional[int]=None,
                   stride:Optional[int]=1):
    """Return UDA movie object for given shot, camera and frame range
    
    :param shot: MAST-U shot number
    :type shot: int
    :param camera: MAST-U camera 3-letter diagnostic code (e.g. rir, rbb etc.)
    :type camera: str
    :param n_start: Frame number of first frame to load
    :type n_start: int
    :param n_end: Frame number of last frame to load
    :type n_end: int
    :param stride: interval between frames to be loaded
    :return: UDA movie object?
    """
    if not isinstance(camera, str) or len(camera) != 3:
        raise ValueError(f'canera argument shoudl be MAST-U camera 3-letter diagnostic code (e.g. rir, rbb etc.)')

    command_str = f'NEWIPX::read(filename=/net/fuslsc/data/MAST_Data/{shot}/LATEST/{camera}0{shot}.ipx)'
    if n_start is not None:
        command_str += f', first={n_start}'
    if n_end is not None:
        command_str += f', last={n_end}'
    if stride not in (None, 1):
        command_str += f', stride={stride}'

    # Read file
    vid = client.get(command_str, '')

    all_times = vid.frame_times
    f0 = vid.frames[0]
    f0_time = f0.time
    f0_data = f0.k

    return vid

if __name__ == '__main__':
    n_start, n_end = 100, 110
    vid = get_frames_uda(30378, 'rir', n_start=n_start, n_end=n_end)
    import pdb; pdb.set_trace()
    print(vid)
    print(vid.frames)
    print(vid.time)
    print(vid.k)
    print(dir(vid))
    pass