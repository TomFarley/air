#!/usr/bin/env python

"""


Created: 
"""

import logging
from typing import Union, Iterable, Sequence, Tuple, Optional, Any, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from fire import fire_paths
from fire.misc.utils import make_iterable
from fire.interfaces.uda_utils import import_pyuda
from fire.plugins.movie_plugins.uda import read_movie_meta, get_uda_movie_obj

logger = logging.getLogger(__name__)
logger.propagate = True
logger.setLevel(logging.DEBUG)

pyuda, client = import_pyuda()

FILTER_FIELDS = ('height', 'width', 'date_time', 'exposure', 'view',
                 'n_frames', 'orientation', 'pre_exp', 'file_format', 'left', 'top', 'trigger')
ADDITIONAL_FIELDS = ('strobe', 'taps', 'hbin', 'vbin', 'filter', 'gain', 'is_color', 'lens', 'offset', )

def get_movie_meta(camera, shot):

    movie_meta = client.get_images(camera, shot, header_only=True)

    return movie_meta

def get_ipx_meta_data_for_pulse_range(camera, shot_list):
    missing_data_shots = []
    data = pd.DataFrame(columns=FILTER_FIELDS)
    data.index.name = 'shot'
    logger.info(f'Attempting to read movie meta deta for {len(shot_list)} shots '
                 f'({np.min(shot_list)}-{np.max(shot_list)}) for camera "{camera}"')
    for shot in shot_list:
        logger.debug(f'Attempting to read movie meta deta for camera "{camera}", shot {shot}')
        # movie_meta1 = get_movie_meta(camera, shot)
        try:
            movie_meta = read_movie_meta(shot, camera=camera)
        except IOError as e:
            logger.warning(f'Failed to read movie meta deta for camera "{camera}", shot {shot}')
            missing_data_shots.append(shot)
        else:
            logger.info(f'Read movie meta deta for camera "{camera}", shot {shot}')
            for field in FILTER_FIELDS:
                data.loc[shot, field] = movie_meta['ipx_header'][field]
    data = data.sort_index()
    n_missing = len(missing_data_shots)
    percentage = 0 if n_missing == 0 else n_missing/(shot_end-shot_start)
    logger.info(f'Failed to read movie meta data for {n_missing}/{(shot_end-shot_start)} ({percentage:%}) shots: '
                f'{missing_data_shots}')
    return data

def format_meta_data_file_path(camera, shot_list, path='{fire}/../tmp/',
                       fn_pattern='meta_data-{camera}-p{start}_{end}_{n}.xlsx'):
    if fn_pattern is None:
        fn_pattern = 'meta_data-{camera}-p{start}_{end}_{n}.xlsx'
    path = Path(path.format(fire=fire_paths['root'])).resolve()
    fn = fn_pattern.format(camera=camera, start=np.min(shot_list), end=np.max(shot_list), n=len(shot_list))
    path_fn = path / fn
    return path_fn

def save_meta_data_set(data, path_fn):
    data.to_excel(path_fn)
    logger.info(f'Wrote meta data to file: {path_fn}')

def load_meta_data_set(path_fn):
    if path_fn.is_file():
        data = pd.read_excel(path_fn)
        logger.info(f'Read meta data to file: {path_fn}')
    else:
        logger.info(f'Existing meta data file does not exist: {path_fn}')
        data = None

    return data

def filter_equal(df, value):
    mask = df == value
    return mask

def filter_dataframe(data, col, filter_type, filter_options):
    filters = dict(equal=filter_equal)
    if filter_type == 'equal':
        func = filter_equal
    else:
        raise ValueError(f'Filter type "{filter_type}" not recognised')

    mask = func(data[col], **filter_options)
    return mask

def apply_filter(data, filter_criteria=None):
    if filter_criteria is None:
        return data
    mask_keep = data['width'] >= 0
    for col, filters in filter_criteria.items():
        for filter_type, filter_options in filters.items():
            mask = filter_dataframe(data, col, filter_type, filter_options)
            mask_keep &= mask
    data = data.loc[mask_keep]
    return data

def get_shot_list_meta_data(camera, shot_list,
                            path='{fire}/../tmp/', fn_pattern=None):
    path_fn = format_meta_data_file_path(camera, shot_list, path=path, fn_pattern=fn_pattern)
    data = load_meta_data_set(path_fn)
    if data is None:
        data = get_ipx_meta_data_for_pulse_range(camera, shot_list)
        if len(data) > 3:
            save_meta_data_set(data, path_fn)
    return data



def filter_pulse_list(camera, shot_list, filter_criteria=None,
                      path='{fire}/../tmp/', fn_pattern='{camera}_{start}-{end}.xlsx'):
    shot_list = np.array(shot_list, dtype=int)
    data = get_shot_list_meta_data(camera, shot_list=shot_list,
                                   path='{fire}/../tmp/', fn_pattern=None)
    print(data)
    data = apply_filter(data, filter_criteria)
    print(f'\nFitler criteria: {filter_criteria}')
    print('\nFiltered data:')
    print(data)
    return data

def plot_param_pulse_variation(data, params):
    params = make_iterable(params)
    n_params = len(params)
    fig, axes = plt.subplots(nrows=n_params, sharex=True)

    for param, ax in zip(params, axes):
        ax.plot(data['shot'], data[param], ls='-', marker='o', ms=5)
        ax.set_ylabel(param)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    camera = 'rir'
    # camera = 'rit'

    # vid = client.get_images('rir', 29976, header_only=False)
    # shot_start, shot_end = 26500, 30300
    # shot_start, shot_end = 29936, 29936

    shot_centre, shot_range = 23586, 40
    shot_start, shot_end = shot_centre-shot_range/2, shot_centre+shot_range/2

    step = 1
    # step = 50

    shot_list = np.arange(shot_start, shot_end+1, step)
    # shot_list = np.array([26957, 27228, 27232, 27642, 27880, 29936])

    filter_criteria = {
        # 'height': {'equal': {'value': 256}},
        # 'width': {'equal': {'value': 256}},
        # 'height': {'equal': {'value': 32}},
        # 'n_frames': {'equal': {'value': 625}},
    }
    data = filter_pulse_list(camera=camera, shot_list=shot_list, filter_criteria=filter_criteria)

    params = ['width', 'height', 'top', 'left']
    plot_param_pulse_variation(data, params)
    pass