#!/usr/bin/env python

""" 
Author: T. Farley
"""

import logging
from typing import Dict, Iterable, Optional
from copy import copy

import numpy as np
import xarray as xr

# fileConfig('../logging_config.ini')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

try:
    import pyuda
    client = pyuda.Client()
except ImportError as e:
    logger.warning(f'Failed to import pyuda. ')
    pyuda = False

movie_plugin_name = 'uda'
plugin_info = {'description': 'This plugin reads movie data from UDA (Universal Data Access)',
               'arg_name_mapping': {'camera': 'camera'}}

uda_ipx_header_fields = ('board_temp', 'camera', 'ccd_temp', 'datetime', 'depth', 'exposure', 'filter', 'frame_times',
                         'gain', 'hbin', 'height', 'is_color', 'left', 'lens', 'n_frames', 'offset', 'preexp', 'shot',
                         'taps', 'top', 'vbin', 'view', 'width')
# uda_ipx_header_fields = ('board_temp', 'camera', 'ccd_temp', 'datetime', 'depth', 'exposure', 'filter', 'frame_times',
#                          'gain', 'hbin', 'height', 'is_color', 'left', 'lens', 'n_frames', 'offset', 'preexp', 'shot',
#                          'taps', 'top', 'vbin', 'view', 'width')
# uda_ipx_header_fields += ('ID', 'size', 'codec', 'date_time', 'trigger', 'orient', 'color', 'hBin',
#                           'right', 'vBin', 'bottom', 'offset_0', 'offset_1', 'gain_0', 'gain_1', 'preExp', 'strobe')

def get_uda_movie_obj(pulse: int, camera: str, n_start:Optional[int]=None, n_end:Optional[int]=None,
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
    :return: UDA movie object?
    """
    if not isinstance(camera, str) or len(camera) != 3:
        raise ValueError(f'Camera argument should be MAST-U camera 3-letter diagnostic code (e.g. rir, rbb etc.)')

    command_str_base = f'NEWIPX::read(filename=/net/fuslsc/data/MAST_Data/{pulse}/LATEST/{camera}0{pulse}.ipx'
    command_str = copy(command_str_base)
    if (n_start is not None) and (n_start == n_end):
        command_str += f', frame={n_start}'
    else:
        if n_start is not None:
            command_str += f', first={n_start}'
        if n_end is not None:
            command_str += f', last={n_end}'
        if stride not in (None, 1):
            command_str += f', stride={stride}'
    command_str += ')'
    # Read file
    try:
        vid = client.get(command_str, '')
    except Exception as e:
        if n_end is not None:
            # Try getting video object without frame arguments
            vid = get_uda_movie_obj(pulse, camera)
            if n_end > vid.n_frames:
                raise ValueError(f'Attempted to read movie (n_frames={vid.n_frames}) with invalid frame range: '
                                 f'{command_str}')
        logger.error(f'Failed to read data from uda with command: {command_str}')
        raise e
    return vid

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
    video = get_uda_movie_obj(pulse, camera, n_start=n_start, n_end=n_end, stride=stride)
    ipx_header = {}
    for key in uda_ipx_header_fields:
        try:
            ipx_header[key] = getattr(video, key)
        except AttributeError as e:
            logger.warning(f'UDA video object does not have attribute: {key}')
    if len(ipx_header) == 0:
        raise ValueError(f'UDA video object does not contain any of the required meta data fields: '
                         f'{uda_ipx_header_fields}')
    ipx_header['bottom'] = ipx_header['top'] - ipx_header['height']  # TODO: check - not +
    ipx_header['right'] = ipx_header['left'] + ipx_header['width']

    last_frame = video.n_frames - 1
    if n_start is None:
        n_start = 0
    if n_end is None:
        n_end = last_frame
    times = video.frame_times

    movie_meta = {'movie_format': '.ipx'}
    movie_meta['n_frames'] = ipx_header['n_frames']
    movie_meta['frame_range'] = np.array([n_start, n_end])
    movie_meta['t_range'] = np.array([times[n_start], times[n_end]])
    movie_meta['frame_shape'] = (video.height, video.width)
    movie_meta['fps'] = (video.n_frames - 1) / np.ptp(times)
    movie_meta['lens'] = ipx_header['lens']
    movie_meta['exposure'] = ipx_header['exposure']
    movie_meta['ipx_header'] = ipx_header
    movie_meta['ipx_header'] = ipx_header
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
    video = get_uda_movie_obj(pulse, camera, n_start=n_start, n_end=n_end, stride=stride)
    if n_start is None:
        n_start = 0
    if n_end is None:
        n_end = video.n_frames - 1
    if stride is None:
        stride = 1
    frame_nos = np.arange(n_start, n_end+1, stride)

    # Allocate memory for frames
    frame_data = np.zeros((len(frame_nos), video.height, video.width))

    frame_times = video.frame_times[n_start:n_end+1:stride]

    for n, frame in enumerate(video.frames):
        frame_data[n, :, :] = frame.k

    if transforms is not None:
        raise NotImplementedError

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