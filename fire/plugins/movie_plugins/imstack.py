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
import matplotlib.pyplot as plt

from pyIpx.movieReader import imstackReader
from fire.interfaces.interfaces import json_load
from fire.plugins.movie_plugins.ipx import get_detector_window_from_ipx_header

logging.basicConfig()
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

movie_plugin_name = 'imstack'
plugin_info = {'description': "This plugin reads folders of image files ('png','jpg','bmp','tif') as a movie"}

def get_freia_imstack_path(pulse, camera):
    """Return path to imstack movie directory on UKAEA freia cluster

    :param pulse: Shot/pulse number or string name for synthetic movie data
    :param camera: Name of camera to analyse (unique name of camera or diagnostic code)
    :return: Path to imstack directory
    """
    pulse = str(pulse)
    imstack_path_fn = f"/projects/SOL/Data/Cameras/SA1/{pulse}/{camera}/C001H001S0001/"
    return imstack_path_fn

def read_movie_meta(path: Union[str, Path], transforms: Iterable[str]=()) -> dict:
    """Read frame data from imstack image directory ('png','jpg','bmp','tif')

    :param path: Path to imstack image directory
    :type path: str, Path
    :param transforms: List of of strings describing transformations to apply to frame data. Options are:
                        'reverse_x', 'reverse_y', 'transpose'
    :type transforms: list
    :return: Dictionary of ipx file information
    :type: dict
    """
    # Read file header and first frame
    vid = imstackReader(directory=path)
    imstack_header = vid.file_header
    ret, frame0, frame_header0 = vid.read(transforms=transforms)
    last_frame_ind = len(vid.file_list) - 1
    image_format = imstack_header['image_format']
    bit_depth = vid.get('EffectiveBitDepth') if vid.get('EffectiveBitDepth') is not None else np.nan

    # Last frame doesn't always load, so work backwards to last successfully loaded frame
    ret = False
    while not ret:
        # Read last frame
        vid._current_index = last_frame_ind
        # vid.set_frame_number(last_frame)
        ret, frame_end, frame_header_end = vid.read(transforms=transforms)
        if not ret:
            # File closes when it fails to load a frame, so re-open
            vid = imstackReader(directory=path)
            last_frame_ind -= 1
    vid.release()

    # Collect summary of ipx file meta data
    # file_header['ipx_version'] = vid.ipx_type
    movie_meta = {'movie_format': image_format}
    movie_meta['n_frames'] = imstack_header['TotalFrame']
    movie_meta['frame_range'] = np.array([0, last_frame_ind])
    t_range = (frame_header0['time_stamp'], frame_header_end['time_stamp'])
    if t_range[0] is not None:
        movie_meta['t_range'] = np.array([float(frame_header0['time_stamp']), float(frame_header_end['time_stamp'])])
    else:
        movie_meta['t_range'] = np.array([np.nan, np.nan])
    image_shape = np.array(frame0.shape)
    movie_meta['image_shape'] = image_shape
    movie_meta['fps'] = (last_frame_ind) / np.ptp(movie_meta['t_range']) if t_range[0] is not None else np.nan
    movie_meta['exposure'] = np.nan   # imstack_header['exposure']
    movie_meta['bit_depth'] = bit_depth
    movie_meta['lens'] = imstack_header['lens'] if 'lens' in imstack_header else None
    # TODO: Add filter name?

    # TODO: Move derived fields to common function for all movie plugins: image_shape, fps, t_range
    # TODO: Check ipx field 'top' follows image/calcam conventions
    movie_meta['width'] = image_shape[1]
    movie_meta['height'] = image_shape[0]
    movie_meta['detector_window'] = get_detector_window_from_ipx_header(movie_meta, plugin='imstack', fn=path)

    imstack_header['imstack_filenames'] = vid.file_list
    movie_meta['imstack_header'] = imstack_header

    # If present, also read meta data from json file with images
    movie_meta_json = json_load(Path(path)/'movie_meta_data.json', raise_on_filenotfound=False)
    if isinstance(movie_meta_json, dict):
        frame_times = movie_meta_json.pop('frame_times', None)
        movie_meta.update(movie_meta_json)
        if frame_times is not None:
            movie_meta['t_range'] = np.array([np.min(frame_times), np.max(frame_times)])
    else:
        logger.warning(f'Imstack movie does not have a meta data json file: {path}')

    return movie_meta

def read_movie_data(path: Union[str, Path],
                    n_start:Optional[int]=None, n_end:Optional[int]=None, stride:Optional[int]=1,
                    frame_numbers: Optional[Union[Iterable, int]]=None,
                    transforms: Optional[Iterable[str]]=(), grayscale: bool=True) -> Tuple[np.ndarray, np.ndarray,
                                                                                     np.ndarray]:
    """Read frame data from imstack image directory ('png','jpg','bmp','tif').

    :param path: Path to imstack directory
    :type path: str, Path
    :param frame_numbers: Frame numbers to read (should be monotonically increasing)
    :type frame_numbers: Iterable[int]
    :param transforms: List of of strings describing transformations to apply to frame data. Options are:
                        'reverse_x', 'reverse_y', 'transpose'
    :type transforms: Optional[Iterable[str]]
    :return: frame_nos, times, data_frames,
    :type: (np.array, np.array ,np.ndarray)
    """
    if transforms is None:
        transforms = ()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Ipx file not found: {path_fn}')
    vid = imstackReader(directory=path)
    imstack_header = vid.file_header
    ret, frame0, frame_header0 = vid.read(transforms=transforms)
    n_frames_movie = int(imstack_header['TotalFrame'])
    if frame0.ndim != 3:
        # If images are not returned with three RGB channels, ignore grayscale conversions checks
        grayscale = False
    elif frame0.shape[2] != 3:
        raise ValueError(f'Unexpected shape for imstack frame data: {frame0.shape}')
    if grayscale and (not np.all(frame0[:, :, 0] == frame0[:, :, 1])):
        logger.warning(f'Reading imstack movie data as grayscale (taking R channel) despite RGB channels not being '
                       f'equal. Consider setting grayscale=False.')

    if frame_numbers is None:
        if (n_start is not None) and (n_end is not None):
            frame_numbers = np.arange(n_start, n_end + 1, stride, dtype=int)
        else:
            frame_numbers = np.arange(n_frames_movie, dtype=int)
    elif isinstance(frame_numbers, numbers.Number):
        frame_numbers = np.array([frame_numbers])
    else:
        frame_numbers = np.array(frame_numbers)
    frame_numbers[frame_numbers < 0] += n_frames_movie
    if any((frame_numbers >= n_frames_movie)):
        raise ValueError(f'Requested frame numbers outside of movie range: '
                         f'{frame_nos[(frame_nos >= vid.file_header["numFrames"])]}')
    if any(np.fmod(frame_numbers, 1) > 1e-5):
        raise ValueError(f'Fractional frame numbers requested from ipx file: {frame_nos}')
    # Allocate memory for frames
    frame_shape = frame0.shape if (not grayscale) else frame0.shape[0:2]
    frame_data = np.zeros((len(frame_numbers), *frame_shape), dtype=frame0.dtype)
    frame_times = np.zeros_like(frame_numbers, dtype=float)

    # To efficiently read the video the frames should be loaded in monotonically increasing order
    frame_numbers = np.sort(frame_numbers).astype(int)
    n, n_end = frame_numbers[0], frame_numbers[-1]
    n_frames_load = len(frame_numbers)

    i_data = 0
    vid.set_frame_number(n)
    logger.debug(f'Reading {n_frames_load} frames from imstack movie over range n=[{n}, {n_end}]')
    n_log = 0
    n_step_log = 400
    while n <= n_end:
        if n in frame_numbers:
            # frames are read with 16 bit dynamic range, but values are 10 bit!
            ret, frame, header = vid.read(transforms=transforms)
            if grayscale:
                # Take only red RGB channel - assume all are same (warning above)
                frame = frame[:, :, 0]
            frame_data[i_data, ...] = frame
            frame_times[i_data] = header['time_stamp']
            i_data += 1
        elif n > n_frames_movie:
            logger.warning('n={} outside ipx movie frame range'.format(n))
            break
        else:
            # Increment vid frame number without reading data
            vid.advance_frame()
        n += 1
        if n // n_step_log > n_log:
            logger.debug(f'n = {n} ({i_data/n_frames_load:0.0%})')
            n_log += 1

    vid.release()

    # If present, also read meta data from json file with images
    movie_meta_json = json_load(Path(path)/'movie_meta_data.json', raise_on_filenotfound=False)
    if isinstance(movie_meta_json, dict) and ('frame_times' in movie_meta_json):
        frame_times = np.array(movie_meta_json['frame_times'])

    return frame_numbers, frame_times, frame_data


if __name__ == '__main__':
    # directory = Path('/projects/SOL/Data/Cameras/SA1/29496/C001H001S0001/').resolve()
    directory = Path('/home/tfarley/data/movies/mast_u/50000/rir/').resolve()
    # frame_nos = np.arange(100, 202)
    frame_nos = None
    meta_data = read_movie_meta(directory)
    frame_nos, frame_times, frame_data = read_movie_data(directory, frame_numbers=frame_nos)
    print(meta_data)
    plt.imshow(frame_data[0])
    pass