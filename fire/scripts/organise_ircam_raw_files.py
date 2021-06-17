#!/usr/bin/env python

"""


Created: 
"""

import logging
from typing import Union, Iterable, Sequence, Tuple, Optional, Any, Dict
from pathlib import Path
from copy import copy
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from fire.interfaces import io_utils, io_basic
from fire.misc.utils import make_iterable

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logger.propagate = False


def complete_meta_data_dict(meta_data_dict, n_frames=None, image_shape=None, replace=False):
    fps = meta_data_dict['fps']
    t_before_pulse = meta_data_dict['t_before_pulse']  # 1e-1
    period = 1 / fps

    if n_frames is None:
        n_frames = meta_data_dict['n_frames']
    if image_shape is None:
        image_shape = meta_data_dict['image_shape']

    # n_frames = len(frame_data)
    # image_shape = list(frame_data.shape[1:])
    detector_window = [0, 0] + list(image_shape)[::-1]
    frame_numbers = np.arange(n_frames).tolist()

    # Make sure frame frame times are centred around frame at t=0
    # TODO: Apply frame rate correction?
    frame_times = list(np.arange(0, -t_before_pulse, -period)[::-1])
    frame_times = frame_times + list(np.arange(period, (n_frames - len(frame_times) + 1) * period, period))

    t_range = [min(frame_times), max(frame_times)]
    frame_range = [min(frame_numbers), max(frame_numbers)]

    dict_out = copy(meta_data_dict)
    dict_out.update(dict(n_frames=n_frames, image_shape=image_shape, detector_window=detector_window,
                        frame_period=period, lens=25e-3, bit_depth=14, t_range=t_range, frame_range=frame_range,
                                    exposure=0.25e-3,  frame_numbers=frame_numbers, frame_times=frame_times,
                                    t_before_pulse=t_before_pulse))
    if not replace:
        dict_out.update(meta_data_dict)
    return dict_out

def generate_ipx_file_from_ircam_raw(path_fn_raw, path_fn_ipx, meta_data_dict, verbose=True):
    from mastvideo import write_ipx_file, IpxHeader, IpxSensor, SensorType, ImageEncoding
    from fire.interfaces.camera_data_formats import read_ircam_raw_int16_sequence_file
    from PIL import Image
    # from ccfepyutils.mast_data.get_data import get_session_log_data
    # import pyuda
    # client = pyuda.Client()

    frame_numbers, data_movie = read_ircam_raw_int16_sequence_file(path_fn_raw, flip_y=True, transpose=False)
    print(f'Read IRCAM raw file {path_fn_raw}')

    n_frames, height, width = tuple(data_movie.shape)
    image_shape = (height, width)

    pulse = meta_data_dict['shot']
    camera = meta_data_dict['camera']

    meta_data_dict = complete_meta_data_dict(meta_data_dict, n_frames=n_frames, image_shape=image_shape)

    times = meta_data_dict['frame_times']
    nuc_frame = data_movie[1]
    frames_ndarray = [frame - nuc_frame for frame in data_movie]
    frames = [Image.fromarray(frame-nuc_frame, mode='I;16') for frame in data_movie]  # PIL images
    # frames = [frame.convert('I;16') for frame in frames]

    # exec(f'import pyuda; client = pyuda.Client(); date_time = client.get_shot_date_time({pulse})')

    # fill in some dummy fields
    header = IpxHeader(
        shot=pulse,
        date_time='<placeholder>',

        camera='IRCAM_Velox81kL_0102',
        view='HL04_A-tangential',
        lens='25 mm',
        trigger=-np.abs(meta_data_dict['t_before_pulse']),
        exposure=int(meta_data_dict['exposure']*1e6),

        num_frames=n_frames,
        frame_width=width,
        frame_height=height,
        depth=14,
    )

    sensor = IpxSensor(
        type=SensorType.MONO,
    )

    path_fn_ipx = Path(str(path_fn_ipx).format(**meta_data_dict)).expanduser()

    with write_ipx_file(
            path_fn_ipx, header, sensor, version=1,
            encoding=ImageEncoding.JPEG2K,
    ) as ipx:
        # write out the frames
        for time, frame in zip(times, frames):
            ipx.write_frame(time, frame)

    message = f'Wrote ipx file: "{path_fn_ipx}"'
    logger.debug(message)
    if verbose:
        print(message)


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
    from fire.interfaces.io_basic import json_dump

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
                             meta=None, camera_settings=None, n_files=None):
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

    if n_files is None:
        n_files = len(files)

    for i, (keys, fn0) in enumerate(reversed(files.items())):
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

        # Copy file from T drive to local archive
        io_basic.copy_file(src, dest, mkdir_dest=True)

        nframes, shape = get_ircam_raw_int_nframes_and_shape(src)

        fn_meta_out = fn_meta.format(**meta)
        generate_json_meta_data_file_for_ircam_raw(dest.parent, fn_meta_out, nframes, image_shape=shape,
                                                   meta_data_dict=camera_settings)

        # generate_ipx_file_from_ircam_raw(dest, meta_data_dict=camera_settings)
        if len(files_filtered) == n_files:
            logger.info(f'Stopped copying after {n_files} files')
            break
    logger.info(f'Copied raw movie files and generated json meta data for {len(files_filtered)} pulses: '
                f'{list(files_filtered.keys())}')

def copy_raw_files_from_tdrive(today=False, n_files=None):
    pulse_whitelist = None
    camera_settings = dict(camera='rit', fps=400, exposure=0.25e-3, lens=25e-3, t_before_pulse=100e-3)
    # fn_in = 'MASTU_LWIR_HL04A-(\d+).RAW'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/Ops_20210128/'
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
    # path_in = '/home/tfarley/ccfepc/T/tfarley/Ops_20210309/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/Ops_20210325/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/Ops_20210326/'
    # fn_in = '(\d+).raw'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/Ops_20210329/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/Ops_20210429/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/Ops_20210430/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/Ops_20210504/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/Ops_20210505/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/Ops_20210507/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/Ops_20210510/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/Ops_20210511/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/Ops_20210512/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/Ops_20210513/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/2021-05-18/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/2021-05-19/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/2021-05-20/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/2021-05-21/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/2021-05-25/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/2021-05-26/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/2021-05-27/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/RIT/2021-05-28/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/RIT/2021-06-02/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/RIT/2021-06-03/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/RIT/2021-06-04/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/RIT/2021-06-15/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/RIT/2021-06-16/'
    path_in = '/home/tfarley/ccfepc/T/tfarley/RIT/2021-06-17/'

    if today:
        path_in = f'/home/tfarley/ccfepc/T/tfarley/RIT/{datetime.now().strftime("%Y-%m-%d")}/'

    # TODO: Extract camera settings meta data from spreadsheet


    path_out = '~/data/movies/mast_u/{pulse}/{camera}/'

    try:
        organise_ircam_raw_files(path_in=path_in, fn_in=fn_in, path_out=path_out, camera_settings=camera_settings,
                                 pulse_whitelist=pulse_whitelist, n_files=n_files)
    except OSError as e:
        logger.exception(f'Failed to copy raw IRCAM files from: {path_in}')
    pass

def convert_raw_files_archive_to_ipx():
    path_root = Path('~/data/movies/mast_u/').expanduser()
    pulses = [p.name for p in path_root.glob('*')]
    print(pulses)
    for pulse in pulses:
        path = path_root / f'{pulse}/rit/'

        fn_raw = f'rit_{pulse}.raw'
        fn_meta = f'rit_{pulse}_meta.json'
        fn_ipx = f'rit0{pulse}.ipx'

        meta_data_dict = dict(io_basic.json_load(fn_meta, path=path))
        meta_data_dict['shot'] = int(pulse)
        if meta_data_dict['fps'] != 400:  # When fps was set to 430 is was actually still aprox 400
            meta_data_dict['fps'] = 400
            meta_data_dict = complete_meta_data_dict(meta_data_dict, replace=True)  # Update frame times
            # TODO: Get frame times from trigger signal?

        generate_ipx_file_from_ircam_raw(path/fn_raw, path/fn_ipx, meta_data_dict=meta_data_dict)

if __name__ == '__main__':
    copy_raw_files_from_tdrive()
    # convert_raw_files_archive_to_ipx()