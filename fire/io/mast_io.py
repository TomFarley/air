#!/usr/bin/env python

"""
Functions for loading and interacting with IR camera data from the MAST tokamak (2000-2013).

Created: 09-10-2019
"""

from typing import Union, Iterable
from pathlib import Path

import numpy as np
import xarray as xr

from pyIpx.movieReader import ipxReader


def get_ipx_meta_data(path_fn: Union[str, Path], transforms: Iterable[str]=()) -> dict:
    """Read frame data from MAST IPX movie file format.

    :param path_fn: Path to IPX movie file
    :type path_fn: str, Path
    :param transforms: List of of strings describing transformations to apply to frame data. Options are:
                        'reverse_x', 'reverse_y', 'transpose'
    :type transforms: list
    :return: Dictionary of ipx file information
    :type: dict
    """
    ipx_meta_data = {'movie_format': '.ipx'}
    n = 0
    step = 100

    # Establish how many frames in movie - first pass in large steps
    vid = ipxReader(filename=path_fn)
    file_header = vid.file_header
    ret, frame0, frame_header0 = vid.read(transforms=transforms)
    while ret:
        n += step
        vid.set_frame_number(n)
        ret, frame, frame_header = vid.read(transforms=transforms)
    n_end = n

    # Establish how many frames in movie - second pass one frame at a time
    vid = ipxReader(filename=path_fn)
    n -= step
    vid.set_frame_number(n)
    ret = True
    while ret:
        n += 1
        ret, frame, frame_header = vid.read(transforms=transforms)  # auto advances vid frame number
    # TODO: Fix problem with seeking to final frame   -2 => -1
    n -= 2  # Go to last frame that returned True

    vid = ipxReader(filename=path_fn)
    vid.set_frame_number(n)
    ret, frame, frame_header = vid.read(transforms=transforms)
    vid.release()

    # Collect summary of ipx file meta data
    ipx_meta_data['ipx_header'] = file_header
    ipx_meta_data['frame_range'] = [0, n]
    ipx_meta_data['t_range'] = [frame_header0['time_stamp'], frame_header['time_stamp']]
    ipx_meta_data['frame_shape'] = frame0.shape
    ipx_meta_data['fps'] = (n + 1) / (frame_header['time_stamp'] - frame_header0['time_stamp'])
    return ipx_meta_data

def get_ipx_frames(ipx_path: Union[str, Path], transforms: Iterable[str]=()) -> np.ndarray:
    """Read frame data from MAST IPX movie file format.

    :param ipx_path: Path to IPX movie file
    :type ipx_path: str, Path
    :param transforms: List of of strings describing transformations to apply to frame data. Options are:
                        'reverse_x', 'reverse_y', 'transpose'
    :type transforms: list
    :return: 3D numpy array of movie frame data
    :type: np.ndarray
    """
    frames = np.zeros((10,10))
    ipx_path = Path(ipx_path)
    assert ipx_path.exists()
    vid = ipxReader(ipx_path, transforms)
    return frames


if __name__ == '__main__':
    ipx_path = Path('test_data/mast/')
    ipx_fn = 'rir030378.ipx'
    ipx_path_fn = ipx_path / ipx_fn
    ipx_meta_data = get_ipx_meta_data(ipx_path_fn)
    print(ipx_meta_data)