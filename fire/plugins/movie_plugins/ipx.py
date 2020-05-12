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

from pyIpx.movieReader import ipxReader

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

movie_plugin_name = 'ipx'
plugin_info = {'description': 'This plugin reads IPX1/2 format MAST movie files'}

def get_freia_ipx_path(pulse, camera):
    """Return path to ipx file on UKAEA freia cluster

    :param pulse: Shot/pulse number or string name for synthetic movie data
    :param camera: Name of camera to analyse (unique name of camera or diagnostic code)
    :return: Path to ipx files
    """
    pulse = str(pulse)
    ipx_path_fn = f"/net/fuslsa/data/MAST_IMAGES/0{pulse[0:2]}/{pulse}/{camera}0{pulse}.ipx"
    return ipx_path_fn

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
    vid = ipxReader(filename=path_fn)
    ipx_header = vid.file_header
    ipx_header = convert_ipx_header_to_uda_conventions(ipx_header)
    ret, frame0, frame_header0 = vid.read(transforms=transforms)
    last_frame = ipx_header['n_frames'] - 1

    # Last frame doesn't always load, so work backwards to last successfully loaded frame
    ret = False
    while not ret:
        # Read last frame
        vid.set_frame_number(last_frame)
        ret, frame_end, frame_header_end = vid.read(transforms=transforms)
        if not ret:
            # File closes when it fails to load a frame, so re-open
            vid = ipxReader(filename=path_fn)
            last_frame -= 1
    vid.release()

    # Collect summary of ipx file meta data
    # file_header['ipx_version'] = vid.ipx_type
    movie_meta = {'movie_format': '.ipx'}
    movie_meta['n_frames'] = ipx_header['n_frames']
    movie_meta['frame_range'] = np.array([0, last_frame])
    movie_meta['t_range'] = np.array([float(frame_header0['time_stamp']), float(frame_header_end['time_stamp'])])
    movie_meta['frame_shape'] = frame0.shape
    movie_meta['fps'] = (last_frame) / np.ptp(movie_meta['t_range'])
    movie_meta['exposure'] = ipx_header['exposure']
    movie_meta['bit_depth'] = ipx_header['depth']
    movie_meta['lens'] = ipx_header['lens'] if 'lens' in ipx_header else None
    # TODO: Add filter name?

    movie_meta['ipx_header'] = ipx_header
    return movie_meta

def convert_ipx_header_to_uda_conventions(header: dict) -> dict:
    """

    :param header: Ipx header dict with each parameter a separate scalar value
    :return: Reformatted header dict
    """
    # TODO: Make generic fulctions for use in all plutins: rename_dict_keys, convert_types, modify_values
    header = copy(header)  # Prevent changes to pyIpx movieReader attribute still used for reading video
    key_map = {'numFrames': 'n_frames', 'color': 'is_color', 'ID': 'ipx_version', 'hBin': 'hbin', 'vBin': 'vbin',
               'date_time': 'datetime', 'preExp': 'preexp'}
    for old_key, new_key in key_map.items():
        try:
            header[new_key] = header.pop(old_key)
            logger.debug(f'Renamed ipx header parameter from "{old_key}" to "{new_key}".')
        except KeyError as e:
            logger.warning(f'Could not rename ipx header parameter to "{new_key}" as paremeter "{old_key}" not found.')
    try:
        header['gain'] = [header.pop('gain_0'), header.pop('gain_1')]
    except KeyError as e:
        logger.warning(f'Could not reformat ipx header parameter "gain" to array as scalar params not found')
    try:
        header['offset'] = [header.pop('offset_0'), header.pop('offset_1')]
    except KeyError as e:
        logger.warning(f'Could not reformat ipx header parameter "offset" to array as scalar params not found')

    # header['bottom'] = header.pop('top') - header.pop('height')  # TODO: check - not +
    # header['right'] = header.pop('left') + header.pop('width')
    return header

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
    path_fn = Path(path_fn)
    if not path_fn.exists():
        raise FileNotFoundError(f'Ipx file not found: {path_fn}')
    vid = ipxReader(filename=path_fn)
    ipx_header = vid.file_header
    n_frames_movie = ipx_header['numFrames']
    if frame_nos is None:
        frame_nos = np.arange(n_frames_movie, dtype=int)
    elif isinstance(frame_nos, numbers.Number):
        frame_nos = np.array([frame_nos])
    else:
        frame_nos = np.array(frame_nos)
    frame_nos[frame_nos < 0] += n_frames_movie
    if any((frame_nos >= n_frames_movie)):
        raise ValueError(f'Requested frame numbers outside of movie range: '
                         f'{frame_nos[(frame_nos >= vid.file_header["numFrames"])]}')
    if any(np.fmod(frame_nos, 1) > 1e-5):
        raise ValueError(f'Fractional frame numbers requested from ipx file: {frame_nos}')
    # Allocate memory for frames
    frame_data = np.zeros((len(frame_nos), ipx_header['height'], ipx_header['width']))
    frame_times = np.zeros_like(frame_nos, dtype=float)

    # To efficiently read the video the frames should be loaded in monotonically increasing order
    frame_nos = np.sort(frame_nos).astype(int)
    n, n_end = frame_nos[0], frame_nos[-1]

    i_data = 0
    vid.set_frame_number(n)
    while n <= n_end:
        if n in frame_nos:
            # frames are read with 16 bit dynamic range, but values are 10 bit!
            ret, frame, header = vid.read(transforms=transforms)
            frame_data[i_data, :, :] = frame
            frame_times[i_data] = header['time_stamp']
            i_data += 1
        elif n > n_frames_movie:
            logger.warning('n={} outside ipx movie frame range'.format(n))
            break
        else:
            # Increment vid frame number without reading data
            vid._skip_frame()
        n += 1
    vid.release()

    return frame_nos, frame_times, frame_data


if __name__ == '__main__':
    ipx_path = Path('../../tests/test_data/mast/').resolve()
    ipx_fn = 'rir030378.ipx'
    ipx_path_fn = ipx_path / ipx_fn
    meta_data = read_movie_meta(ipx_path_fn)
    frame_nos, frame_times, frame_data = read_movie_data(ipx_path_fn)
    print(meta_data)