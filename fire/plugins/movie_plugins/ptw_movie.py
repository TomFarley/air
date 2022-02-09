#!/usr/bin/env python

"""


Created: Tom Farley & Fabio Federici, 27-04-2021
"""

import logging
from typing import Union, Iterable, Sequence, Tuple, Optional, Any, Dict
from pathlib import Path
from collections import namedtuple
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from fire.interfaces.io_basic import json_load

logger = logging.getLogger(__name__)


logging.basicConfig()
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

MovieData = namedtuple('movie_plugin_frame_data', ['frame_numbers', 'frame_times', 'frame_data'])

movie_plugin_name = 'ats_movie'
plugin_info = {'description': "This plugin reads ptw movie files output by the FLIR Altair software"}

def read_movie_meta(path_fn: Union[str, Path], raise_on_missing_meta=True, verbose=True) -> dict:
    """Read meta data for ptw movie file (eg exported from the FLIR Altair software).

    :param path_fn: Path to ats movie file
    :type path_fn: str, Path
    :return: Dictionary of meta data from accompanying json meta data file information
    :type: dict
    """
    from fire.interfaces.ats_flir_movie_reader import ats_to_dict, read_ats_file_header
    from fire.plugins.movie_plugins.ipx_standard import (get_detector_window_from_ipx_header,
                                                         check_ipx_detector_window_meta_data)

    if verbose:
        print(f'{datetime.now()}: Reading ats meta data: {path_fn}')

    path_fn = Path(path_fn)
    # TODO: Make more efficient by not reading in frame data here
    # movie_dict = read_ats_file_header(path_fn.parent, path_fn.name)
    movie_dict = ats_to_dict(path_fn.parent, path_fn.name, n_frames_read=2)

    movie_meta = dict(movie_format='.ptw')

    # If present, also read meta data from json file with movie file
    path_fn = Path(path_fn)
    path = path_fn.parent if path_fn.is_file() else path_fn
    path_fn_meta = path / 'rir_meta.json'  # TODO: Remove hardcoded rir
    if not path_fn_meta.exists() and path_fn.is_file():
        path_fn_meta = path / (str(path_fn.stem) + '_meta.json')

    movie_meta_json = json_load(path_fn_meta, raise_on_filenotfound=False, lists_to_arrays=True)

    if isinstance(movie_meta_json, list):
        movie_meta_json = dict(movie_meta_json)  # Saved as json list in order to save ints, floats etc

    if isinstance(movie_meta_json, dict):
        movie_meta.update(movie_meta_json)
    else:
        message = f'FLIR .ptw movie does not have a meta data json file: {path}'
        if raise_on_missing_meta:
            raise IOError(message)
        else:
            logger.warning(message)



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

    if len(set(np.diff(movie_dict['frame_counter']))) > 1:
        # TODO: Remove end frames with frame count jumps?
        if verbose:
            logger.warning(f'Frame counter misses frames: {set(np.diff(movie_dict["frame_counter"]))}')
    movie_meta['n_frames'] = len(movie_dict['frame_counter'])

    movie_meta['frame_range'] = np.array([0, movie_meta['n_frames']-1])
    movie_meta['frame_numbers'] = np.arange(movie_meta['frame_range'][0], movie_meta['frame_range'][1]+1)

    movie_meta['image_shape'] = np.array([movie_meta['height'], movie_meta['width']])

    if 'clock_period' in movie_meta:
        movie_meta['fps'] = 1 / movie_meta['clock_period']
    else:
        movie_meta['fps'] = movie_meta['n_frames'] / (np.max(movie_meta['frame_times'])
                                                      - np.min(movie_meta['frame_times']))
    if 'clock_start' in movie_meta:
        movie_meta['frame_times'] = movie_meta['clock_start'] + 1/movie_meta['fps'] * movie_meta['frame_numbers']
    else:
        # time_of_measurement is int not float
        movie_meta['frame_times'] = movie_dict['time_of_measurement']  # TODO: Refine

    movie_meta['t_range'] = np.array([np.min(movie_meta['frame_times']), np.max(movie_meta['frame_times'])])

    check_ipx_detector_window_meta_data(movie_meta, plugin='ats', fn=path_fn, modify_inplace=True)  # Complete missing fields
    movie_meta['detector_window'] = get_detector_window_from_ipx_header(movie_meta)  # left, top, width, height

    # TODO: Rename meta data fields to standard

    if verbose:
        print(f'{datetime.now()}: Finished reading ats meta data')

    return movie_meta


def read_movie_data(path_fn: Union[str, Path], write_ipx=False, raise_on_missing_meta=True) -> MovieData:
    """Read frame data for ptw movie file (eg exported from the FLIR Altair software).

    :param path_fn: Path to ats movie file
    :type path_fn: str, Path
    :return: Dictionary of meta data from accompanying json meta data file information
    :type: dict
    """
    from fire.interfaces.ptw_flir_movie_reader import ptw_to_dict

    t0 = datetime.now()
    print(f'{t0}: Reading ats movie data: {path_fn}')

    movie_meta = read_movie_meta(path_fn=path_fn, raise_on_missing_meta=raise_on_missing_meta, verbose=False)

    path_fn = Path(path_fn)
    movie_dict = ptw_to_dict(path_fn)

    frame_data = movie_dict['data']
    frame_times = movie_meta['frame_times']
    # frame_numbers = movie_dict['frame_counter']
    frame_numbers = movie_meta['frame_numbers']

    t1 = datetime.now()
    print(f'{t1}: Finished reading ats movie data ({(t1-t0).total_seconds()} s)')

    if write_ipx:
        convert_aps_data_to_ipx(movie_dict, movie_meta, overwrite=True)

    return MovieData(frame_numbers, frame_times, frame_data)

def convert_aps_data_to_ipx(movie_dict, movie_meta, path_ipx_archive='~/data/movies/mast_u/', overwrite=True):
    # from fire.scripts.organise_ircam_raw_files import generate_ipx_file_from_flir_ats_movie
    from fire.plugins.movie_plugins.ipx import write_ipx_with_mastmovie
    path_ipx_archive = Path(path_ipx_archive).expanduser()

    pulse = int(movie_meta["pulse"])
    camera = movie_meta["camera"]
    path_fn_ipx = path_ipx_archive / f'{pulse}/{diag_tag_raw}/{diag_tag_raw}0{pulse}.ipx'

    frame_data = movie_dict['data']
    frame_times = movie_meta['frame_times']
    movie_meta['frame_times'] = frame_times
    movie_meta['ID'] = 'IPX 02'

    if overwrite or (not path_fn_ipx.is_file()):
        pil_frames = write_ipx_with_mastmovie(path_fn_ipx, frame_data, header_dict=movie_meta,
                                              apply_nuc=False, create_path=True, verbose=True)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # path = '/home/ffederic/work/irvb/vacuum_chamber_testing/Aug23_2018'
    # file_ats = 'irvb_sample-000001.ats'

    # path = Path('/home/tfarley/data/movies/mast_u/FLIR_data-tfarley/')
    # file_ats = 'flir_air_sequence_16-04-2021.ats'

    # path = Path('/home/tfarley/data/movies/mast_u/rir_ats_files/2021-08-19')
    # file_ats = '044749.ats'
    # path = Path('/home/tfarley/data/movies/mast_u/rir_ats_files/2021-08-17')
    # file_ats = '044684.ats'
    shot = 44677
    path = Path('/home/tfarley/data/movies/diagnostic_pc_transfer/rir/2021-08-13/')

    file_ats = f'0{shot}.ats'

    print(f'Reading FLIR ats movie (may be slow): {path}/{file_ats}')
    movie_meta = read_movie_meta(path / file_ats)
    movie_data = read_movie_data(path / file_ats)

    print(movie_meta)

    plt.figure(f'ats movie {file_ats}')
    plt.imshow(movie_data.frame_data[100])
    plt.show()
    pass