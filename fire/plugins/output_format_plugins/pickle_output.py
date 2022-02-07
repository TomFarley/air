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

import fire.interfaces.io_basic

from fire.misc.utils import filter_kwargs
from fire.interfaces import io_utils

# logger = logging.getLogger(__name__)
logger = logging.getLogger('fire.pickle_output')
# logger.setLevel(logging.DEBUG)

# ===================== PLUGIN MODULE ATTRIBUTES =====================
# Required:
output_format_plugin_name = 'pickle_output'
# Optional:
output_filename_format = '{diag_tag_analysed}{shot:06d}.p'  # Filename of output
output_path_format = '~/{fire_user_dir}/pickle_output_archive/{diag_tag_analysed}/'  # Path to save output
# See bottom of file for function aliases
# ====================================================================

not_set = object()

def write_processed_ir_to_pickle_output_file(path_data, image_data, path_names,
                                         variable_names_path=None, variable_names_time=None, variable_names_image=None,
                                         header_info=None, device_info=None, meta_data=None,
                                         fn_output=None, path_output=None, filter_output=False):
    """"""
    from fire.interfaces.io_basic import pickle_dump

    if meta_data is None:
        meta_data = {}

    if fn_output is None:
        fn_output = output_filename_format
    if path_output is None:
        path_output = output_path_format

    path = Path(str(path_output).format(**meta_data)).expanduser()
    fn = str(fn_output).format(**meta_data)
    path_fn = path / fn

    if not path.is_dir():
        fire.interfaces.io_basic.mkdir(path, depth=3)

    if filter_output:
        try:
            image_data_out = image_data[variable_names_image]
            path_data_out = path_data[variable_names_path]  # TODO: variable_names_time?
        except Exception as e:
            logger.exception(e)
            # TODO: Improve error handling for missing variable names
    else:
        image_data_out = image_data
        path_data_out = path_data

    data = dict(path_data=path_data_out, image_data=image_data_out, path_names=path_names,
                variable_names_path=variable_names_path, variable_names_time=variable_names_time,
                variable_names_image=variable_names_image, header_info=header_info, device_info=device_info,
                meta_data=meta_data)

    pickled_keys = list(data.keys())
    data['_pickled_keys'] = pickled_keys

    status = pickle_dump(data, path_fn, remove_custom_classes=False, verbose=False, raise_exceptions=False)

    if (status is False) or (not path_fn.is_file()):
        # raise FileNotFoundError(f'UDA output file does not exist: {path_fn}')
        logger.exception(f'pickle output file does not exist: {path_fn}')
        success = False
    else:
        success = True
        logger.info(f'Pickle output file written to: {path_fn}')

    return dict(success=success, path_fn=path_fn)

def read_processed_ir_to_pickle_output_file(camera, pulse, machine='mast_u',
                                            path_archive='{fire_user_dir}/pickle_output_archive/{diag_tag_analysed}/',
                                            fn_format='{diag_tag_analysed}{pulse:06d}.p', meta_data=None):
    from fire.interfaces.io_basic import pickle_load
    from fire.interfaces import uda_utils

    if meta_data is None:
        meta_data = {}
        
    config = fire.get_user_fire_config()
    meta_data.update(config.config_groups['fire_paths'])
    # meta_data.setdefault('fire_user_dir', '~/fire/')

    diag_tag_raw = camera
    meta_args = dict(camera=camera, pulse=pulse, shot=pulse, machine=machine,
                     diag_tag_raw=diag_tag_raw, diag_tag_analysed=uda_utils.get_analysed_diagnostic_tag(diag_tag_raw))

    path = Path(str(path_archive).format(**meta_args, **meta_data)).expanduser()
    fn = str(fn_format).format(**meta_args, **meta_data)
    path_fn = path / fn

    try:
        out = pickle_load(path_fn)
    except FileNotFoundError as e:
        raise e

    try:
        out_unpacked = [out[key] for key in out['_pickled_keys']]
    except KeyError as e:
        out_unpacked = [out[key] for key in out]

    out = (out, out_unpacked)

    return out

# ================== PLUGIN MODULE FUNCTION ALIASES ==================
write_output_file = write_processed_ir_to_pickle_output_file
read_output_file = read_processed_ir_to_pickle_output_file
# ====================================================================

if __name__ == '__main__':
    pass