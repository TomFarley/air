#!/usr/bin/env python
"""
The `interfaces` module contains functions for interfacing with other codes and files.
"""

from typing import Union, Iterable, Optional
from pathlib import Path

import numpy as np
import xarray as xr

from fire.interfaces.calcam import get_calcam_calib_path_fn

def load_user_defaults():
    """Return user's default settings

    :return: Dict of user's default settings
    :rtype: dict
    """
    user_defaults = {'machine': 'MAST', 'camera': 'rit', 'pulse': 23586}
    return user_defaults

def identify_input_files(pulse, camera, machine):
    """Return dict of paths to files needed for IR analysis

    :param pulse: Shot/pulse number or string name for synthetic movie data
    :param camera: Name of camera to analyse (unique name of camera or diagnostic code)
    :param machine: Tokamak that the data originates from
    :return: Dict of filepaths
    """
    files = {}
    files['calcam_calib'] = get_calcam_calib_path_fn(pulse, camera, machine)

    # TODO: check path characters are safe (see setpy datafile code)

    return files

def read_movie_meta_data(pulse, camera, machine):
    if machine == 'MAST':
        from fire.interfaces.ipx import read_movie_meta_ipx
        from fire.interfaces.uda import read_movie_meta_uda
        meta_data = read_movie_meta_uda(pulse, camera)
    else:
        raise NotImplementedError(f'Camera data acquisition not implemented for machine: "{machine}"')
    return meta_data

def read_movie_data(pulse, camera, machine):
    if machine == 'MAST':
        from fire.interfaces.ipx import read_movie_data_ipx
        from fire.interfaces.uda import read_movie_data_uda
        frame_nos, frame_times, frame_data = read_movie_data_uda(pulse, camera)
    else:
        raise NotImplementedError(f'Camera data acquisition not implemented for machine: "{machine}"')
    return frame_nos, frame_times, frame_data

def generate_pulse_id_strings(pulse, camera, machine, pass_no, lens, t_int):
    """Return standardised ID strings used for consistency in filenames and data labels

    :param pulse: Shot/pulse number or string name for synthetic movie data
    :param camera: Name of camera to analyse (unique name of camera or diagnostic code)
    :param machine: Tokamak that the data originates from
    :param pass_no: Scheduler pass number
    :param lens: Lens on camera
    :param t_int: Integration time of camera in seconds
    :return: Dict of ID strings
    """
    pulse_id = f'{machine}-{pulse}'
    camera_id = f'{pulse_id}-{camera}'

    # calcam_id = f'{machine}-{camera}-{calib_date}-{pass_no}'

    id_strings = {'pulse_id': pulse_id,
                  'camera_id': camera_id}

    return id_strings

def generate_camera_id_strings(camera_id_strings, lens, t_int):
    """Return standardised ID strings used for consistency in filenames and data labels

    :param pulse: Shot/pulse number or string name for synthetic movie data
    :param camera: Name of camera to analyse (unique name of camera or diagnostic code)
    :param machine: Tokamak that the data originates from
    :param pass_no: Scheduler pass number
    :param lens: Lens on camera
    :param t_int: Integration time of camera in seconds
    :return: Dict of ID strings
    """
    camera_id = camera_id_strings['camera_id']
    lens_id = f'{camera_id}-{lens}'
    t_int_id = f'{lens_id}-{t_int}'

    id_strings = {'lens_id': lens_id,
                    't_int_id': t_int_id, }
    return id_strings

def generate_frame_id_strings(camera_id_strings, frame_no, frame_time):
    """Return ID strings for specific analysis frame

    :param camera_id_string: Strings output by generate_camera_id_strings()
    :param pulse: Shot/pulse number or string name for synthetic movie data
    :param camera: Name of camera to analyse (unique name of camera or diagnostic code)
    :param machine: Tokamak that the data originates from
    :param pass_no: Scheduler pass number
    :param lens: Lens on camera
    :param t_int: Integration time of camera in seconds
    :return:
    """
    camera_id = camera_id_strings['camera_id']
    pulse_id = camera_id_strings['pulse_id']

    frame_id = f'{camera_id}-{frame_no}'
    time_id = f'{pulse_id}-{time}'

    id_strings = {'frame_id': frame_id,
                    'time_id': time_id, }
    return id_strings


def check_frame_range(meta_data, frames=None, start_frame=None, end_frame=None, nframes_user=None, frame_stride=1):
    raise NotImplementedError