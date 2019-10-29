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

uda_ipx_header_fields = {'board_temp', 'camera', 'ccd_temp', 'datetime', 'depth', 'exposure', 'filter', 'frame_times',
                         'gain', 'hbin', 'height', 'is_color', 'left', 'lens', 'n_frames', 'offset', 'preexp', 'shot',
                         'taps', 'top', 'vbin', 'view', 'width'}
def return_true():
    return True

def get_uda_movie_obj(shot: int, camera: str, n_start:Optional[int]=None, n_end:Optional[int]=None,
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
        raise ValueError(f'Camera argument should be MAST-U camera 3-letter diagnostic code (e.g. rir, rbb etc.)')

    command_str = f'NEWIPX::read(filename=/net/fuslsc/data/MAST_Data/{shot}/LATEST/{camera}0{shot}.ipx'
    if n_start is not None:
        command_str += f', first={n_start}'
    if n_end is not None:
        command_str += f', last={n_end}'
    if stride not in (None, 1):
        command_str += f', stride={stride}'
    command_str += ')'
    # Read file
    vid = client.get(command_str, '')
    return vid

def read_movie_meta_uda(shot: int, camera: str, n_start:Optional[int]=None, n_end:Optional[int]=None,
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
    :return: Movie meta data
    """
    video = get_uda_movie_obj(shot, camera, n_start=n_start, n_end=n_end, stride=stride)
    ipx_header = {}
    for key in uda_ipx_header_fields:
        ipx_header[key] = getattr(video, key)
    last_frame = video.n_frames - 1
    times = vid.frame_times

    movie_meta = {'movie_format': '.ipx'}
    movie_meta['ipx_header'] = ipx_header
    movie_meta['frame_range'] = np.array([0, last_frame])
    movie_meta['t_range'] = np.array([times[0], times[-1]])
    movie_meta['frame_shape'] = (video.height, video.width)
    movie_meta['fps'] = (last_frame) / np.ptp(movie_meta['t_range'])
    return movie_meta

def read_movie_data_uda(shot: int, camera: str, n_start:Optional[int]=None, n_end:Optional[int]=None,
                  stride:Optional[int]=1, transforms: Iterable[str]=None):
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
    :type stride: int
    :param transforms: List of of strings describing transformations to apply to frame data. Options are:
                    'reverse_x', 'reverse_y', 'transpose'
    :type transforms: Optional[Iterable[str]]
    :return: UDA movie object?
    """
    video = get_uda_movie_obj(shot, camera, n_start=n_start, n_end=n_end, stride=stride)
    frame_nos = np.arange(n_start, n_end+1, stride)

    # Allocate memory for frames
    frame_data = np.zeros((len(frame_nos), video.height, video.width))

    frame_times = video.frame_times

    for n, frame in enumerate(video.frames):
        frame_data[n, :, :] = frame.k

    if transforms is not None:
        raise NotImplementedError

    return frame_nos, frame_times, frame_data

if __name__ == '__main__':
    shot = 30378
    camera = 'rir'
    # camera = 'air'
    n_start, n_end = 100, 110
    vid = get_uda_movie_obj(shot, camera, n_start=n_start, n_end=n_end)
    # import pdb; pdb.set_trace()
    meta_data = read_movie_meta_uda(shot, camera, n_start, n_end)
    frame_nos, frame_times, frame_data = read_movie_data_uda(shot, camera, n_start, n_end)

    r = client.list(pyuda.ListType.SIGNALS, shot=shot, alias='air')
    signals = {}
    for d in r:
        signals[d.signal_name] = d.description
    print(signals)
    pass