#!/usr/bin/env python

"""


Created: 
"""

import logging, re
from typing import Union, Iterable, Sequence, Tuple, Optional, Any, Dict
from pathlib import Path
from copy import copy
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from fire.interfaces import io_utils, io_basic
from fire.misc.utils import make_iterable, safe_arange
from fire.plotting import plot_tools

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
    width = image_shape[1]
    height = image_shape[0]

    dict_out = copy(meta_data_dict)
    dict_out.update(dict(n_frames=n_frames, image_shape=image_shape, detector_window=detector_window,
                         width=width, height=height, top=0, left=0, right=width, bottom=height,
                        frame_period=period, lens=25e-3, bit_depth=14, t_range=t_range, frame_range=frame_range,
                                    exposure=0.25e-3,  frame_numbers=frame_numbers, frame_times=frame_times,
                                    t_before_pulse=t_before_pulse))
    if not replace:
        dict_out.update(meta_data_dict)
    return dict_out

def generate_ipx_file_from_ircam_raw_movie(path_fn_raw, path_fn_ipx, pulse, verbose=True, plot_check=False):
    # from fire.interfaces.camera_data_formats import read_ircam_raw_int16_sequence_file
    from fire.plugins.movie_plugins.raw_movie import read_movie_data, read_movie_meta
    from fire.plugins.movie_plugins.ipx import write_ipx_with_mastmovie
    from fire.plugins.movie_plugins.ipx import read_movie_data_with_mastmovie, read_movie_data_with_pyipx
    from fire.plugins.movie_plugins.ipx import read_movie_meta_with_mastmovie, read_movie_meta_with_pyipx
    from fire.misc import utils
    from fire.plugins.plugins_movie import MovieReader
    from PIL import Image
    # from ccfepyutils.mast_data.get_data import get_session_log_data
    # import pyuda
    # client = pyuda.Client()

    # movie_reader = MovieReader(plugin_filter=('raw_movie', 'ats_movie'))
    # frame_numbers, frame_times, frame_data = movie_reader.read_movie_data(pulse=pulse, camera='rir', machine='mast_u')
    # meta_data_dict = movie_reader.read_movie_meta_data(pulse=pulse, camera='rir', machine='mast_u')
    meta_data_dict = read_movie_meta(path_fn_raw)
    frame_numbers, frame_times, frame_data = read_movie_data(path_fn_raw)
    print(f'Read IRCAM raw file {path_fn_raw}')

    # frame_data = frame_data - frame_data[1]  # tmp

    meta_data_dict['shot'] = int(pulse)
    if meta_data_dict['fps'] != 400:  # When fps was set to 430 is was actually still aprox 400
        meta_data_dict['fps'] = 400
        meta_data_dict = complete_meta_data_dict(meta_data_dict, replace=True)  # Update frame times
        # TODO: Get frame times from trigger signal?

    meta_data_dict['frame_times'] = frame_times
    # n_frames, height, width = tuple(frame_data.shape)
    # image_shape = (height, width)
    #
    # pulse = meta_data_dict['shot']
    # # camera = meta_data_dict['camera']
    #
    # meta_data_dict = complete_meta_data_dict(meta_data_dict, n_frames=n_frames, image_shape=image_shape)
    #


    # exec(f'import pyuda; client = pyuda.Client(); date_time = client.get_shot_date_time({pulse})')

    # fill in some dummy fields
    # header = dict(
    #     shot=pulse,
    #     date_time='<placeholder>',
    #     camera='IRCAM_Velox81kL_0102',
    #     view='HL04_A-tangential',
    #     lens='25 mm',
    #     trigger=-np.abs(meta_data_dict['t_before_pulse']),
    #     exposure=int(meta_data_dict['exposure']*1e6),
    #     num_frames=n_frames,
    #     frame_width=width,
    #     frame_height=height,
    #     depth=14,
    # )
    pil_frames = write_ipx_with_mastmovie(path_fn_ipx, frame_data, header_dict=meta_data_dict, verbose=True)
    
    if plot_check:
        n = 250
        frame_original = frame_data[n]

        frame_numbers_mastmovie, frame_times_mastmovie, data_mastmovie = read_movie_data_with_mastmovie(path_fn_ipx)
        meta_mastmovie = read_movie_meta_with_mastmovie(path_fn_ipx)
        frame_mastmovie = data_mastmovie[n]

        frame_numbers_pyipx, frame_times_pyipx, data_pyipx = read_movie_data_with_pyipx(path_fn_ipx)
        meta_pyipx = read_movie_meta_with_pyipx(path_fn_ipx)
        frame_pyipx = data_pyipx[n]

        for dic in [meta_data_dict, meta_mastmovie, meta_pyipx]:
            for key in ['frame_times', 'shot']:
                if key in dic:
                    dic.pop(key)
        utils.compare_dict(meta_data_dict, meta_mastmovie)
        utils.compare_dict(meta_data_dict, meta_pyipx)

        print(meta_data_dict)
        print(meta_mastmovie)

        # plt.ion()
        fig, axes, ax_passed = plot_tools.get_fig_ax(ax=None, ax_grid_dims=(2, 3), sharex=True, sharey=True,
                                                     axes_flatten=True)
        fig.suptitle(f'{pulse}, n={n}')

        ax = axes[0]
        im0 = ax.imshow(frame_original, interpolation='none', origin='upper', cmap='gray')  # , vmin=0, vmax=2**14-1)
        # plt.colorbar(im0)
        ax.set_title(f'Original raw')

        ax = axes[1]
        im1 = ax.imshow(frame_mastmovie, interpolation='none', origin='upper', cmap='gray')  # , vmin=0, vmax=2**14-1)
        # plt.colorbar(im1)
        ax.set_title(f'Mastvideo output')

        ax = axes[2]
        im2 = ax.imshow(frame_pyipx, interpolation='none', origin='upper', cmap='gray')  # , vmin=0, vmax=2**14-1)
        # plt.colorbar(im1)
        ax.set_title(f'pyIpx output')

        ax = axes[3]
        im2 = ax.imshow(frame_original-frame_original, interpolation='none', origin='upper', cmap='gray')
        # plt.colorbar(im1)
        ax.set_title(f'Original-Original')

        ax = axes[4]
        im2 = ax.imshow(frame_original - frame_mastmovie, interpolation='none', origin='upper', cmap='gray')
        # plt.colorbar(im1)
        ax.set_title(f'Original-mastmovie output')
        print('Origianl - mastmovie', np.max(frame_original-frame_mastmovie), np.mean(frame_original-frame_mastmovie))

        ax = axes[5]
        im2 = ax.imshow(frame_original-frame_pyipx, interpolation='none', origin='upper', cmap='gray')
        # plt.colorbar(im1)
        ax.set_title(f'Original-pyIpx output')
        print('Origianl - pyIpx', np.max(frame_original-frame_pyipx), np.mean(frame_original-frame_pyipx))


        plt.tight_layout()
        plt.show()

        pil_frames[n].show(title=f'PIL native show {pulse}, {n}')
        pass


def generate_ipx_file_from_flir_ats_movie(path_fn_ats, path_fn_ipx, pulse, verbose=True, plot_check=True):
    # from fire.interfaces.camera_data_formats import read_ircam_raw_int16_sequence_file
    from fire.plugins.movie_plugins.ats_movie import read_movie_data, read_movie_meta
    from fire.plugins.movie_plugins.ipx import write_ipx_with_mastmovie
    from fire.plugins.movie_plugins.ipx import read_movie_data as read_movie_data_ipx
    from fire.plugins.movie_plugins.ipx import read_movie_meta as read_movie_meta_ipx

    frame_numbers, frame_times, frame_data = read_movie_data(path_fn_ats)
    meta_data_dict = read_movie_meta(path_fn_ats)
    print(f'Read FLIR ats file {path_fn_ats}')

    # frame_data = frame_data - frame_data[1]  # tmp

    meta_data_dict['shot'] = int(pulse)

    # TODO: Compare to frame times from trigger signal?

    meta_data_dict['frame_times'] = frame_times

    pil_frames = write_ipx_with_mastmovie(path_fn_ipx, frame_data, header_dict=meta_data_dict,
                                          apply_nuc=False, create_path=True, verbose=True)

    if plot_check:
        n = 250
        frame_numbers_out, frame_times_out, data_out = read_movie_data_ipx(path_fn_ipx)
        meta_new = read_movie_meta_ipx(path_fn_ipx)
        frame_new = data_out[n]
        frame_original = frame_data[n]

        meta_data_dict.pop('frame_times');
        meta_data_dict.pop('shot');

        print(meta_data_dict)
        print(meta_new)

        plt.ion()
        fig, (ax0, ax1) = plt.subplots(1, 2, sharex=True, sharey=True)
        fig.suptitle(f'{pulse}, n={n}')
        im0 = ax0.imshow(frame_original, interpolation='none', origin='upper', cmap='gray')  # , vmin=0, vmax=2**14-1)
        # plt.colorbar(im0)
        ax0.set_title(f'Original raw')
        im1 = ax1.imshow(frame_new, interpolation='none', origin='upper', cmap='gray')  # , vmin=0, vmax=2**14-1)
        # plt.colorbar(im1)
        ax1.set_title(f'Mastvideo output')
        plt.tight_layout()
        plt.show()

        pil_frames[n].show(title=f'PIL native show {pulse}, {n}')
        pass

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
    view = meta_data_dict.get('view', 'HL04_A-tangential')
    t_before_pulse = meta_data_dict['t_before_pulse']  # 1e-1
    period = 1/fps

    # n_frames = len(frame_data)
    # image_shape = list(frame_data.shape[1:])
    detector_window = [0, 0] + list(image_shape)[::-1]
    frame_numbers = np.arange(n_frames).tolist()

    # Make sure frame frame times are centred around frame at t=0
    frame_times = list(safe_arange(0, -t_before_pulse, -period)[::-1])
    frame_times = frame_times + list(safe_arange(period, (n_frames-len(frame_times))*period, period))

    t_range = [min(frame_times), max(frame_times)]
    frame_range = [min(frame_numbers), max(frame_numbers)]

    dict_out = dict(n_frames=n_frames, image_shape=image_shape, detector_window=detector_window, frame_period=period,
                    lens=25e-3, bit_depth=14, t_range=t_range, frame_range=frame_range, exposure=0.25e-3,
                    frame_numbers=frame_numbers, frame_times=frame_times, t_before_pulse=t_before_pulse)
    dict_out.update(meta_data_dict)

    list_out = list(dict_out.items())

    assert len(frame_times) == n_frames

    json_dump(list_out, fn, path, overwrite=True)
    logger.info(f'Wrote meta data file to: {path}/{fn}')

def organise_ircam_raw_files(path_in='/home/tfarley/data/movies/diagnostic_pc_transfer/{date}/',
                             fn_in='(\d+).RAW', fn_in_group_keys=('pulse',), date='today',
                             path_out='~/data/movies/mast_u/{pulse}/{camera}/', fn_raw_out='{camera}_{pulse}.raw',
                             fn_meta='{camera}_{pulse}_meta.json', fn_ipx_format='rit0{pulse}.ipx',
                             pulse_whitelist=None, pulse_blacklist=None,
                             meta=None, camera_settings=None, n_files=None, write_ipx=True, overwrite_ipx=True):
    from fire.interfaces.io_utils import filter_files_in_dir
    from fire.interfaces.camera_data_formats import read_ircam_raw_int16_sequence_file, get_ircam_raw_int_nframes_and_shape

    today_str = datetime.now().strftime('%Y-%m-%d')
    date_str = today_str if (date == 'today') else date
    path_in = Path(str(path_in).format(today=today_str, date=date_str))

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

    print(f'Located {len(files)} raw movie files in folder ""{path_in}":\n{list(files.keys())}"')

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

        fn_raw_src = (Path(path_in) / fn0).expanduser()
        fn_raw_dest = (Path(path_out.format(**meta)) / fn_raw_out.format(**meta)).expanduser()

        # Copy file from temparary to local structured archive
        io_basic.copy_file(fn_raw_src, fn_raw_dest, mkdir_dest=True)

        nframes, shape = get_ircam_raw_int_nframes_and_shape(fn_raw_src)

        fn_meta_out = fn_meta.format(**meta)
        generate_json_meta_data_file_for_ircam_raw(fn_raw_dest.parent, fn_meta_out, nframes, image_shape=shape,
                                                   meta_data_dict=camera_settings)
        if write_ipx:
            # Create ipx file in same directory
            fn_ipx = fn_raw_dest.with_name(fn_ipx_format.format(pulse=keys))
            if overwrite_ipx or (not fn_ipx.is_file()):
                generate_ipx_file_from_ircam_raw_movie(fn_raw_dest, fn_ipx, pulse=keys, plot_check=True)
            # generate_ipx_file_from_ircam_raw(dest, meta_data_dict=camera_settings)

        if len(files_filtered) == n_files:
            logger.info(f'Stopped copying after {n_files} files')
            break
    logger.info(f'Copied raw movie files and generated json meta data for {len(files_filtered)} pulses: '
                f'{list(files_filtered.keys())}')

def copy_raw_files_from_staging_area(date='today', n_files=None, write_ipx=True, overwrite_ipx=True,
                                     path_archive='/home/tfarley/data/movies/diagnostic_pc_transfer/rit/{date}/',
                                     path_out='~/data/movies/mast_u/{pulse}/{camera}/'):
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

    # camera_settings = dict(camera='rit', fps=400, exposure=0.25e-3, lens=25e-3, t_before_pulse=100e-3)
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
    # path_in = '/home/tfarley/ccfepc/T/tfarley/RIT/2021-06-17/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/RIT/2021-06-18/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/RIT/2021-06-22/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/RIT/2021-06-23/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/RIT/2021-06-24/'
    # path_in = '/home/tfarley/ccfepc/T/tfarley/RIT/2021-06-25/'

    # path_in = '/home/tfarley/ccfepc/T/tfarley/RIT/2021-06-30/'

    camera_settings = dict(camera='rit', fps=400, exposure=0.25e-3, lens=25e-3, t_before_pulse=1e-1)
    # path_in = '/home/tfarley/data/movies/diagnostic_pc_transfer/2021-06-29/'
    # path_in = '/home/tfarley/data/movies/diagnostic_pc_transfer/2021-06-30/'
    # path_in = '/home/tfarley/data/movies/diagnostic_pc_transfer/2021-07-01/'
    # path_in = '/home/tfarley/data/movies/diagnostic_pc_transfer/2021-07-05/'
    # path_in = '/home/tfarley/data/movies/diagnostic_pc_transfer/2021-07-06/'
    # path_in = '/home/tfarley/data/movies/diagnostic_pc_transfer/2021-07-07/'
    # path_in = '/home/tfarley/data/movies/diagnostic_pc_transfer/2021-07-08/'
    # path_in = '/home/tfarley/data/movies/diagnostic_pc_transfer/2021-07-09/'
    # path_in = '/home/tfarley/data/movies/diagnostic_pc_transfer/2021-07-13/'
    # path_in = '/home/tfarley/data/movies/diagnostic_pc_transfer/2021-07-27/'
    # path_in = '/home/tfarley/data/movies/diagnostic_pc_transfer/2021-07-28/'
    # path_in = '/home/tfarley/data/movies/diagnostic_pc_transfer/2021-07-29/'
    # path_in = '/home/tfarley/data/movies/diagnostic_pc_transfer/2021-08-03/'
    # path_in = '/home/tfarley/data/movies/diagnostic_pc_transfer/2021-08-04/'
    # path_in = '/home/tfarley/data/movies/diagnostic_pc_transfer/2021-08-11/'
    # path_in = '/home/tfarley/data/movies/diagnostic_pc_transfer/2021-08-12/'
    # path_in = '/home/tfarley/data/movies/diagnostic_pc_transfer/2021-08-13/'
    # path_in = '/home/tfarley/data/movies/diagnostic_pc_transfer/2021-08-18/'
    # path_in = '/home/tfarley/data/movies/diagnostic_pc_transfer/2021-08-19/'
    # path_in = '/home/tfarley/data/movies/diagnostic_pc_transfer/2021-08-20/'
    # path_in = '/home/tfarley/data/movies/diagnostic_pc_transfer/2021-08-24/'
    # path_in = '/home/tfarley/data/movies/diagnostic_pc_transfer/2021-08-26/'
    # path_in = '/home/tfarley/data/movies/diagnostic_pc_transfer/2021-09-01/'
    # path_in = '/home/tfarley/data/movies/diagnostic_pc_transfer/2021-09-08/'

    if path_archive is not None:
        today_str = datetime.now().strftime('%Y-%m-%d')

        if (date == 'today'):
            date_str = today_str
        elif isinstance(date, int):
            date_str = (datetime.today()-timedelta(days=-date)).strftime('%Y-%m-%d')
        else:
            date_str = date
        print(f'Date: {date_str}')
        path_in = Path(str(path_archive).format(today=today_str, date=date_str))

    # TODO: Extract camera settings meta data from spreadsheet

    try:
        organise_ircam_raw_files(path_in=path_in, fn_in=fn_in, path_out=path_out, camera_settings=camera_settings,
                                 pulse_whitelist=pulse_whitelist, n_files=n_files, write_ipx=write_ipx,
                                 overwrite_ipx=overwrite_ipx)
    except OSError as e:
        logger.exception(f'Failed to copy raw IRCAM files from: {path_in}')
    pass

def convert_raw_files_archive_to_ipx(pulses=None, path=None):
    path_root = Path('~/data/movies/mast_u/').expanduser()
    if pulses is None:
        pulses = pulses[:1]  # tmp
        pulses = list(sorted([p.name for p in path_root.glob('[!.]*')]))
        pulses = [44677, 44683]
        pulses = [44613]
    if path is not None:
        path = Path(path)

    print(pulses)

    for pulse in pulses:
        path = path_root / f'{pulse}/rit/'

        fn_raw = f'rit_{pulse}.raw'
        fn_meta = f'rit_{pulse}_meta.json'
        fn_ipx = f'rit0{pulse}.ipx'

        if not (path/fn_raw).is_file():
            logger.warning(f'Raw file does not exist for pulse {pulse}: {fn_raw}')
            continue

        generate_ipx_file_from_ircam_raw_movie(path / fn_raw, path / fn_ipx, pulse=pulse)


def convert_ats_files_archive_to_ipx(pulses=None, path_in=None, date=None, copy_ats_file=False, fn_ats='0{pulse}.ats',
                                     fn_meta='/home/tfarley/data/movies/mast_u/rir_ats_files/rir_meta.json'):
    from fire.interfaces.interfaces import locate_file
    from fire.interfaces.io_utils import filter_files, filter_files_in_dir
    path_root = Path('~/data/movies/mast_u/').expanduser()

    success, failed = [], []

    if (date is None) or (date == 'today'):
        date = datetime.now().strftime('%Y-%m-%d')

    if path_in is not None:
        path_in = Path(str(path_in).format(date=date)).expanduser()
    else:
        path_in = path_root / date

    print(f'Path_in: {path_in}')
    io_basic.copy_file(fn_meta, path_in, mkdir_dest=False, verbose=True, overwrite=True)

    if pulses is None:
        pulses = list(reversed(sorted([p.stem for p in path_in.glob('[!.]*')])))
        # pulses = [44677, 44683]
        # pulses = [44613]
        # pulses = pulses[:1]  # tmp

    print(pulses)

    for pulse in pulses:
        try:
            pulse = re.match('.*0?(\d{5})', str(pulse)).groups()[0]
            pulse = int(pulse)
        except Exception as e:
            print(f'Skipping file stem "{pulse}" in {path_in}')
            continue
        # path_in = path_root / f'rir_ats_files/'  # (\d4-\d2-\d2)/'
        path_out = path_root / f'{pulse}/rir/'

        fn_ats_formatted = fn_ats.format(pulse=pulse, shot=pulse)
        # fn_ats = f'rir_{pulse}.ats'
        # for path in path_in.glob('*'):
        #     fn_ats = path / f'rir_{pulse}.ats'

        fn_meta = f'rir_meta.json'
        fn_ipx = f'rir0{pulse}.ipx'

        if not (path_in / fn_ats_formatted).is_file():
            logger.warning(f'FLIR ats file does not exist for pulse {pulse}: {fn_ats}')
            continue

        path_fn_ats = path_in / fn_ats_formatted
        path_fn_ipx = path_out / fn_ipx
        try:
            generate_ipx_file_from_flir_ats_movie(path_fn_ats, path_fn_ipx, pulse=pulse, plot_check=False)
        except IndexError as e:
            logger.exception(f'Failed to convert ats file to ipx for shot {pulse}')
            failed.append(pulse)
        else:
            success.append(pulse)

        if copy_ats_file:
            io_basic.copy_file(path_fn_ats, Path(path_fn_ipx).with_suffix('.ats'), overwrite=True, verbose=True)
            io_basic.copy_file(path_fn_ats.with_name(fn_meta), Path(path_fn_ipx).parent, overwrite=True,
                               raise_exceptions=False, verbose=True)
    print(f'Successfully created ipx files for pulses {success}, failed for {failed}')

if __name__ == '__main__':
    # copy_raw_files_from_staging_area(write_ipx=True, date=-3)
    # convert_raw_files_archive_to_ipx(path=Path('~/data/movies/mast_u/44777/rit/').expanduser())

    # path_ats = '~/data/movies/mast_u/rir_ats_files/{date}'
    path_ats = '~/data/movies/diagnostic_pc_transfer/rir/{date}'

    convert_ats_files_archive_to_ipx(path_in=path_ats, copy_ats_file=False,
                                     fn_ats='0{pulse}.ats',
                                     # fn_ats='rir-0{pulse}.ats',
                                     date='today',
                                     # date='2021-08-13',
                                     # pulses=[44677],
                                     # date='2021-05-13',
                                     # pulses=[43952],  # Calibration movie
                                     pulses=[44980],  # Calibration movie
                                     )