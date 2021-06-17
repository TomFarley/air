#!/usr/bin/env python

"""


Created: Tom Farley & Fabio Federici, 27-04-2021
"""

import logging
from typing import Union, Iterable, Sequence, Tuple, Optional, Any, Dict
from pathlib import Path
from collections import namedtuple

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger.propagate = False

logging.basicConfig()
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

MovieData = namedtuple('movie_plugin_frame_data', ['frame_numbers', 'frame_times', 'frame_data'])

movie_plugin_name = 'ats_movie'
plugin_info = {'description': "This plugin reads ats movie files output by the FLIR ResearchIR software"}

def read_movie_meta(path_fn: Union[str, Path], raise_on_missing_meta=True) -> dict:
    """Read meta data for ats movie file (eg exported from the FLIR ResearchIR software).

    :param path_fn: Path to ats movie file
    :type path_fn: str, Path
    :return: Dictionary of meta data from accompanying json meta data file information
    :type: dict
    """
    from fire.interfaces.flir_ats_movie_reader import ats_to_dict
    from fire.plugins.movie_plugins.ipx import get_detector_window_from_ipx_header

    path_fn = Path(path_fn)
    # TODO: Make more efficient by not reading in frame data here
    movie_dict = ats_to_dict(path_fn.parent, path_fn.name)

    movie_meta = dict(movie_format='.ats')

    header_fields = {'digitizer_ID': 'digitizer_ID',
                     'time_of_measurement': 'time_of_measurement',
                     'IntegrationTime': 'exposure',
                     'FrameRate': 'fps',
                     'ExternalTrigger': 'ExternalTrigger',
                     'SensorTemp_0': 'SensorTemp_0',
                     'DetectorTemp': 'DetectorTemp',
                     'width': 'width',
                     'height': 'height',
                     'camera_SN': 'camera_SN',
                     'frame_counter': 'frame_counter'
    }

    movie_meta.update({name: movie_dict[key] for key, name in header_fields.items()})

    frame_numbers = movie_dict['frame_counter']
    movie_meta['n_frames'] = len(frame_numbers)
    frame_times = movie_dict['time_of_measurement']


    movie_meta['frame_range'] = np.array([0, movie_meta['n_frames']])
    movie_meta['t_range'] = np.array([frame_times[0], frame_times[1]])  # TODO: Refine
    movie_meta['image_shape'] = np.array([movie_meta['height'], movie_meta['width']])
    movie_meta['fps'] = movie_meta['n_frames'] / (movie_meta['t_range'][1] - movie_meta['t_range'][0])
    movie_meta['detector_window'] = get_detector_window_from_ipx_header(movie_meta, plugin='ats', fn=path_fn)

    return movie_meta


def read_movie_data(path_fn: Union[str, Path], raise_on_missing_meta=True) -> MovieData:
    """Read frame data for ats movie file (eg exported from the FLIR ResearchIR software).

    :param path_fn: Path to ats movie file
    :type path_fn: str, Path
    :return: Dictionary of meta data from accompanying json meta data file information
    :type: dict
    """
    from fire.interfaces.flir_ats_movie_reader import ats_to_dict

    path_fn = Path(path_fn)
    movie_dict = ats_to_dict(path_fn.parent, path_fn.name)

    frame_data = movie_dict['data']
    frame_times = movie_dict['time_of_measurement']
    frame_numbers = movie_dict['frame_counter']

    return MovieData(frame_numbers, frame_times, frame_data)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # path = '/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018'
    # file_ats = 'irvb_sample-000001.ats'

    path = Path('/home/tfarley/data/movies/mast_u/FLIR_data-tfarley/')
    file_ats = 'flir_air_sequence_16-04-2021.ats'

    print(f'Reading FLIR ats movie (may be slow): {path}/{file_ats}')
    movie_meta = read_movie_meta(path / file_ats)
    movie_data = read_movie_data(path / file_ats)

    print(movie_meta)

    plt.figure(f'ats movie {file_ats}')
    plt.imshow(movie_data.frame_data[10])
    plt.show()
    pass