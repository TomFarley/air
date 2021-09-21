# -*- coding: future_fstrings -*-
#!/usr/bin/env python

"""Plugin functions for FIRE to read movie data from a sever using the Universal Data Access (UDA) library.

Two functions and two module level variables are required for this file to  function as a FIRE movie plugin:
    read_movie_data(pulse, camera, n_start, n_end, stride) -> (frame_nos, frame_times, frame_data)
    read_movie_meta(pulse, camera, n_start, n_end, stride) -> dict with minimum subset of keys:
        {'movie_format', 'n_frames', 'frame_range', 't_range', 'image_shape', 'fps', 'lens', 'exposure', 'bit_depth'}
    movie_plugin_name: str, typically same as module name
    plugin_info: dict, with description and any other useful information or mappings

Author: T. Farley
"""

import logging
from typing import Dict, Iterable, Sequence, Optional
from copy import copy

import numpy as np

from fire.plugins.movie_plugins.ipx_standard import (get_detector_window_from_ipx_header,
                                                         check_ipx_detector_window_meta_data)
from fire.interfaces import uda_utils
from fire.plugins.movie_plugins import ipx_standard

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

UDA_IPX_HEADER_FIELDS = ipx_standard.UDA_IPX_HEADER_FIELDS

try:
    import pyuda, cpyuda
    client = pyuda.Client()
    # pyuda.Client.server = "uda2.hpc.l"  # UDA2 (new office network server). See https://users.mastu.ukaea.uk/sites/default/files/uploads/UDA_data_access.pdf
    # pyuda.Client.port = 56565
except ImportError as e:
    logger.warning(f'Failed to import pyuda. ')
    pyuda = False

movie_plugin_name = 'uda'
plugin_info = {'description': 'This plugin reads movie data from UDA (Universal Data Access)',
               'arg_name_mapping': {'camera': 'camera'}}

use_mast_client = True

def get_uda_movie_obj_legacy(pulse: int, camera: str, n_start:Optional[int]=None, n_end:Optional[int]=None,
                             stride: Optional[int]=1):
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
    except cpyuda.ServerException as e:
        raise IOError(f'Failed to read ipx file with UDA command: {command_str}')
    except Exception as e:
        if n_end is not None:
            # Try getting video object without frame arguments
            vid = get_uda_movie_obj_legacy(pulse, camera)
            if n_end > vid.n_frames:
                raise ValueError(f'Attempted to read movie (n_frames={vid.n_frames}) with invalid frame range: '
                                 f'{command_str}')
        logger.error(f'Failed to read data from uda with command: {command_str}')
        raise e
    return vid


def get_uda_movie_obj(pulse: int, camera: str, n_start: Optional[int] = None, n_end: Optional[int] = None,
                             stride: Optional[int] = 1, use_mast_client=True, try_alternative_client=True):
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
    # TODO: Test other keywords and allow ipx path in place of pulse no?
    uda_module, client = uda_utils.get_uda_client(use_mast_client=use_mast_client, try_alternative=True)

    try:
        vid = client.get_images(camera, pulse, first_frame=n_start, last_frame=n_end, stride=stride,
                                # frame_number=None,
                                header_only=False, rcc_calib_path=None)
    except AttributeError as e:
        message = f'Failed to get uda vid object with {client} (use_mast_client={use_mast_client}): {e}'
        logger.warning(message)
        print(message)
        if try_alternative_client:
            vid = get_uda_movie_obj(pulse, camera, n_start=n_start, n_end=n_end, stride=stride,
                                    use_mast_client=(not use_mast_client), try_alternative_client=False)
        else:
            raise e
    else:
        logger.debug(f'Instantiated video object from uda using client: {client}')
        print(f'Instantiated video object from uda using client: {client}')  # TODO: remove
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
    # TODO: Update to use new function client.get_images(header_only=True)
    # video = get_uda_movie_obj_legacy(pulse, camera, n_start=n_start, n_end=n_end, stride=stride)
    video = get_uda_movie_obj(pulse, camera, n_start=n_start, n_end=n_end, stride=stride)

    ipx_header = {}
    for key in UDA_IPX_HEADER_FIELDS:
        try:
            ipx_header[key] = getattr(video, key)
        except AttributeError as e:
            logger.warning(f'UDA video object does not have attribute: {key}')
    if len(ipx_header) == 0:
        raise ValueError(f'UDA video object does not contain any of the required meta data fields: '
                         f'{UDA_IPX_HEADER_FIELDS}')
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
    movie_meta['t_before_pulse'] = np.abs(ipx_header.get('trigger', movie_meta['t_range'][0]))
    movie_meta['image_shape'] = np.array((video.height, video.width))
    movie_meta['fps'] = (video.n_frames - 1) / np.ptp(times)
    movie_meta['lens'] = ipx_header['lens'] if 'lens' in ipx_header else 'Unknown'
    movie_meta['exposure'] = ipx_header['exposure']
    movie_meta['bit_depth'] = ipx_header['depth']
    # TODO: Add filter name?

    check_ipx_detector_window_meta_data(movie_meta, plugin='uda', fn=path_fn, modify=True)  # Complete missing fields
    movie_meta['detector_window'] = get_detector_window_from_ipx_header(movie_meta)  # left, top, width, height

    movie_meta['ipx_header'] = ipx_header

    return movie_meta

def read_movie_data(pulse: int, camera: str,
                    frame_numbers:Optional[Sequence[int]]=None,
                    n_start:Optional[int]=None, n_end:Optional[int]=None, stride:Optional[int]=1,
                    transforms: Iterable[str]=None):
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
    # video = get_uda_movie_obj_legacy(pulse, camera, n_start=n_start, n_end=n_end, stride=stride)
    video = get_uda_movie_obj(pulse, camera, n_start=n_start, n_end=n_end, stride=stride)

    if frame_numbers is None:
        if n_start is None:
            n_start = 0
        if n_end is None:
            n_end = video.n_frames - 1
        if stride is None:
            stride = 1
    frame_numbers = np.arange(n_start, n_end+1, stride)

    # Allocate memory for frames
    frame_data = np.zeros((len(frame_numbers), video.height, video.width))

    # TODO: allow frame numbers not starting at 0? - see mask for npz
    frame_times = video.frame_times[frame_numbers]

    # NOTE: video object already only contains subset of frames specified in call to uda
    for i in np.arange(len(video.frames)):
        frame_data[i, :, :] = video.frames[i].k

    if (transforms is not None) and (len(transforms) > 0):
        raise NotImplementedError

    return frame_numbers, frame_times, frame_data

if __name__ == '__main__':
    # pulse = 30378
    pulse = 26505
    # pulse = 28623
    camera = 'rir'
    # camera = 'air'
    n_start, n_end = 100, 110
    # vid = get_uda_movie_obj_legacy(pulse, camera, n_start=n_start, n_end=n_end)
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