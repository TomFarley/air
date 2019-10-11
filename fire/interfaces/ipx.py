#!/usr/bin/env python

"""
Functions for loading and interacting with IR camera data from the MAST tokamak (2000-2013).

Created: 09-10-2019
"""

from typing import Union, Iterable, Tuple, Optional
from pathlib import Path

import numpy as np
import xarray as xr

from pyIpx.movieReader import ipxReader

def return_true():
    return True

def get_freia_ipx_path(shot, camera):
    shot = str(shot)
    ipx_path_fn = f"/net/fuslsa/data/MAST_IMAGES/0{shot[0:2]}/{shot}/{camera}0{shot}.ipx"
    return ipx_path_fn

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
    # Read file header and first frame
    vid = ipxReader(filename=path_fn)
    file_header = vid.file_header
    ret, frame0, frame_header0 = vid.read(transforms=transforms)
    last_frame = file_header['numFrames'] - 1

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
    file_header['ipx_version'] = vid.ipx_type
    ipx_meta_data = {'movie_format': '.ipx'}
    ipx_meta_data['ipx_header'] = file_header
    ipx_meta_data['frame_range'] = np.array([0, last_frame])
    ipx_meta_data['t_range'] = np.array([float(frame_header0['time_stamp']), float(frame_header_end['time_stamp'])])
    ipx_meta_data['frame_shape'] = frame0.shape
    ipx_meta_data['fps'] = (last_frame + 1) / np.ptp(ipx_meta_data['t_range'])
    return ipx_meta_data



def get_ipx_frame(ipx_path_fn: Union[str, Path], transforms: Iterable[str]=()) -> Tuple[np.ndarray, dict]:
    """Read frame data from MAST IPX movie file format.

    :param ipx_path_fn: Path to IPX movie file
    :type ipx_path_fn: str, Path
    :param transforms: List of of strings describing transformations to apply to frame data. Options are:
                        'reverse_x', 'reverse_y', 'transpose'
    :type transforms: list
    :return: 3D numpy array of movie frame data
    :type: np.ndarray
    """


def get_ipx_frames(ipx_path_fn: Union[str, Path], frames: Optional[Iterable]=None,
                   transforms: Iterable[str]=()) -> Tuple[bool,dict,np.ndarray]:
    """Read frame data from MAST IPX movie file format.

    :param ipx_path_fn: Path to IPX movie file
    :type ipx_path_fn: str, Path
    :param frames: Frame numbers to read (should be monotonically increasing)
    :type frames: Iterable[int]
    :param transforms: List of of strings describing transformations to apply to frame data. Options are:
                        'reverse_x', 'reverse_y', 'transpose'
    :type transforms: list
    :return: success, header, frames,
    :type: (bool,dict,np.ndarray)
    """
    ipx_path_fn = Path(ipx_path_fn)
    assert ipx_path_fn.exists()
    vid = ipxReader(filename=ipx_path_fn)
    file_header = vid.file_header
    if any((frames > vid.file_header['numFrames'])):
        raise ValueError(f'Requested frame numbers outside of movie range: '
                         f'{frames[(frames > vid.file_header["numFrames"])]}')
    # Allocate memory for frames
    data_frames = np.zeros((len(frames), file_header['shape']))

    # To efficiently read the video the frames should be loaded in monotonically increasing order
    frames = np.sort(frames)
    n, n_end = frames[0], frames[-1]
    vid.set_frame_number(n)
    while n < n_end:
        if n in frames:
            # frames are read with 16 bit dynamic range, but values are 10 bit!
            ret, frame, header = vid.read(transforms=self._transforms)
            data[i_data, :, :] = frame
            self._meta.loc[n, 'set'] = True
            i_data += 1
        elif n > self._movie_meta['frame_range'][1]:
            logger.warning('n={} outside ipx movie frame range'.format(n))
            break
        else:
            # TODO: Increment vid frame number without reading data
            vid._skip_frame()
            # ret, frame, header = vid.read(transforms=self._transforms)
        n += 1
    vid.release()

    return frames


if __name__ == '__main__':
    ipx_path = Path('../../tests/test_data/mast/').resolve()
    ipx_fn = 'rir030378.ipx'
    ipx_path_fn = ipx_path / ipx_fn
    ipx_meta_data = get_ipx_meta_data(ipx_path_fn)
    frames = get_ipx_frames(ipx_path_fn)
    print(ipx_meta_data)