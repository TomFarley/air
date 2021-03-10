#!/usr/bin/env python

"""


Created: 
"""

import logging
from typing import Union, Iterable, Sequence, Tuple, Optional, Any, Dict
from pathlib import Path
from copy import copy

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from fire.interfaces import io_utils
from fire.misc.utils import make_iterable

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logger.propagate = False


def generate_json_meta_data_file_for_ircam_raw(path, fn, n_frames, image_shape, meta_data_dict):
    """
    See movie_meta_required_fields in plugins_movie.py, line ~260:
      ['n_frames', 'frame_range', 't_range', 'fps', 'lens', 'exposure', 'bit_depth', 'image_shape', 'detector_window']

    Args:
        path:
        fn:
        frame_data:
        meta_data_dict:

    Returns:

    """
    from fire.interfaces.interfaces import json_dump

    fps = meta_data_dict['fps']
    t_before_pulse = meta_data_dict['t_before_pulse']  # 1e-1
    period = 1/fps

    # n_frames = len(frame_data)
    # image_shape = list(frame_data.shape[1:])
    detector_window = [0, 0] + list(image_shape)[::-1]
    frame_numbers = np.arange(n_frames).tolist()

    # Make sure frame frame times are centred around frame at t=0
    frame_times = list(np.arange(0, -t_before_pulse, -period)[::-1])
    frame_times = frame_times + list(np.arange(period, (n_frames-len(frame_times)+1)*period, period))

    t_range = [min(frame_times), max(frame_times)]
    frame_range = [min(frame_numbers), max(frame_numbers)]

    dict_out = dict(n_frames=n_frames, image_shape=image_shape, detector_window=detector_window, frame_period=period,
                    lens=25e-3, bit_depth=14, t_range=t_range, frame_range=frame_range, exposure=0.25e-3,
                    frame_numbers=frame_numbers, frame_times=frame_times, t_before_pulse=t_before_pulse)
    dict_out.update(meta_data_dict)

    list_out = list(dict_out.items())

    json_dump(list_out, fn, path, overwrite=True)
    logger.info(f'Wrote meta data file to: {path}/{fn}')

def organise_ircam_raw_files(path_in='/home/tfarley/ccfepc/T/tfarley/Ops_20210130/',
                             fn_in='MASTU_LWIR_HL04A-(\d+).RAW', fn_in_group_keys=('pulse',),
                             path_out='~/data/movies/mast_u/{pulse}/{camera}/', fn_out='{camera}_{pulse}.raw',
                             fn_meta='{camera}_{pulse}_meta.json',
                             pulse_whitelist=None, pulse_blacklist=None,
                             meta=None, camera_settings=None):
    from fire.interfaces.io_utils import filter_files_in_dir
    from fire.interfaces.camera_data_formats import read_ircam_raw_int16_sequence_file, get_ircam_raw_int_nframes_and_shape

    if meta is None:
        meta = {}
    if camera_settings is None:
        camera_settings = {}
    meta = copy(meta)
    meta.update(camera_settings)  # copy camera name etc

    files = filter_files_in_dir(path_in, fn_pattern=fn_in, group_keys=fn_in_group_keys)  # , pulse=pulse_whitelist)
    files_filtered = {}
    for keys, fn0 in files.items():
        if pulse_blacklist is not None:
            if keys in make_iterable(pulse_blacklist):
                continue
        if pulse_whitelist is not None:
            if keys not in make_iterable(pulse_whitelist):
                continue

        files_filtered[keys] = fn0

        kws = dict(zip(make_iterable(fn_in_group_keys), make_iterable(keys)))
        meta.update(kws)

        src = (Path(path_in) / fn0).expanduser()
        dest = (Path(path_out.format(**meta)) / fn_out.format(**meta)).expanduser()

        io_utils.copy_file(src, dest, mkdir_dest=True)

        nframes, shape = get_ircam_raw_int_nframes_and_shape(src)

        fn_meta_out = fn_meta.format(**meta)
        generate_json_meta_data_file_for_ircam_raw(dest.parent, fn_meta_out, nframes, image_shape=shape,
                                                   meta_data_dict=camera_settings)

    logger.info(f'Copied raw movie files and generated json meta data for {len(files_filtered)} pulses: '
                f'{list(files_filtered.keys())}')

if __name__ == '__main__':
    pulse_whitelist = None
    camera_settings = dict(camera='rit', fps=400, exposure=0.25e-3, lens=25e-3, t_before_pulse=100e-3)
    # fn_in = 'MASTU_LWIR_HL04A-(\d+).RAW'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/Ops_20210130/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/Ops_20210203/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/Ops_20210209/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/Ops_20210211/'  # NOTE: t_before_pulse is incorrect for shots before 43331

    # fn_in = 'rit_(\d+).RAW'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/Ops_20210215/'

    fn_in = '(\d+).RAW'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/Ops_20210216/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/Ops_20210218/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/Ops_20210224/'

    # path_in = '/home/tfarley/ccfepc/T/tfarley/Ops_20210225/'
    # camera_settings = dict(camera='rit', fps=430, exposure=0.1e-3, lens=25e-3, t_before_pulse=100e-3)
    # pulse_whitelist = [43547]

    # camera_settings = dict(camera='rit', fps=430, exposure=0.25e-3, lens=25e-3, t_before_pulse=100e-3)
    # path_in = '/home/tfarley/ccfepc/T/tfarley/Ops_20210226/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/Ops_20210227/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/Ops_20210228/'

    camera_settings = dict(camera='rit', fps=400, exposure=0.25e-3, lens=25e-3, t_before_pulse=100e-3)
    # path_in = '/home/tfarley/ccfepc/T/tfarley/Ops_20210301/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/Ops_20210302/'
    path_in = '/home/tfarley/ccfepc/T/tfarley/Ops_20210309/'

    # TODO: Extract camera settings meta data from spreadsheet


    path_out = '~/data/movies/mast_u/{pulse}/{camera}/'

    organise_ircam_raw_files(path_in=path_in, fn_in=fn_in, path_out=path_out, camera_settings=camera_settings,
                             pulse_whitelist=pulse_whitelist)
    pass