#!/usr/bin/env python

"""
Functions for loading and interacting with IR camera data from the MAST tokamak (2000-2013).

Created: 09-10-2019
"""

import logging
from typing import Union, Iterable, Tuple, Optional
from pathlib import Path

import numpy as np
import xarray as xr

from pyIpx.movieReader import ipxReader

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def return_true():
    return True

def get_freia_ipx_path(shot, camera):
    shot = str(shot)
    ipx_path_fn = f"/net/fuslsa/data/MAST_IMAGES/0{shot[0:2]}/{shot}/{camera}0{shot}.ipx"
    return ipx_path_fn

def read_movie_meta_ipx(path_fn: Union[str, Path], transforms: Iterable[str]=()) -> dict:
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
    ret, frame0, frame_header0 = vid.read(transforms=transforms)
    last_frame = ipx_header['numFrames'] - 1

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
    meta_data = {'movie_format': '.ipx'}
    meta_data['ipx_header'] = ipx_header
    meta_data['frame_range'] = np.array([0, last_frame])
    meta_data['t_range'] = np.array([float(frame_header0['time_stamp']), float(frame_header_end['time_stamp'])])
    meta_data['frame_shape'] = frame0.shape
    meta_data['fps'] = (last_frame) / np.ptp(meta_data['t_range'])
    return meta_data


def read_movie_data_ipx(ipx_path_fn: Union[str, Path], frame_nos: Optional[Iterable]=None,
                        transforms: Optional[Iterable[str]]=()) -> Tuple[bool,dict,np.ndarray]:
    """Read frame data from MAST IPX movie file format.

    :param ipx_path_fn: Path to IPX movie file
    :type ipx_path_fn: str, Path
    :param frame_nos: Frame numbers to read (should be monotonically increasing)
    :type frame_nos: Iterable[int]
    :param transforms: List of of strings describing transformations to apply to frame data. Options are:
                        'reverse_x', 'reverse_y', 'transpose'
    :type transforms: Optional[Iterable[str]]
    :return: frame_nos, times, data_frames,
    :type: (np.array, np.array ,np.ndarray)
    """
    ipx_path_fn = Path(ipx_path_fn)
    assert ipx_path_fn.exists()
    vid = ipxReader(filename=ipx_path_fn)
    ipx_header = vid.file_header
    n_frames_movie = ipx_header['numFrames']
    if frame_nos is None:
        frame_nos = np.arange(n_frames_movie)
    if any((frame_nos >= n_frames_movie)):
        raise ValueError(f'Requested frame numbers outside of movie range: '
                         f'{frame_nos[(frame_nos >= vid.file_header["numFrames"])]}')
    # Allocate memory for frames
    frame_data = np.zeros((len(frame_nos), ipx_header['height'], ipx_header['width']))
    frame_times = np.zeros_like(frame_nos, dtype=float)

    # To efficiently read the video the frames should be loaded in monotonically increasing order
    frame_nos = np.sort(frame_nos)
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
    meta_data = read_movie_meta_ipx(ipx_path_fn)
    frame_nos, frame_times, frame_data = read_movie_data_ipx(ipx_path_fn)
    print(meta_data)