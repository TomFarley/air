#!/usr/bin/env python

"""Functions for interfacing with Calcam


Created: 11-10-19
"""

import logging
from typing import Union, Iterable, Tuple, Optional
from pathlib import Path

import numpy as np
import pandas as pd

import calcam
from fire import fire_paths

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

calcam_calib_dir = Path('~/calcam2/calibrations/').expanduser()

def get_calcam_calib_path_fn(pulse: int, camera: str, machine: str):
    """Return path to calcam calibration file for given discharge and camera

    :param pulse: Shot/pulse number
    :param camera: Camera to look up calbration for
    :param machine: Tokamak under analysis
    :return: calib_path_fn
    """
    # TODO: Handle synthetic pulses with non numeric pulse numbers
    machine_inputs_path = fire_paths['input_files'] / f'{machine.lower()}'
    user_inputs_path = fire_paths['user_inputs'] / f'{machine.lower()}'
    calib_lookup_dflt_path_fn =  machine_inputs_path / f'calcam_calibs-{machine.lower()}-{camera.lower()}-defaults.csv'
    calib_lookup_user_path_fn = user_inputs_path / f'calcam_calibs-{machine.lower()}-{camera.lower()}-defaults.csv'

    path_fn = calib_lookup_user_path_fn
    calib_info = lookup_pulse_row_in_csv(path_fn, pulse)
    if isinstance(calib_info, Exception):
        path_fn = calib_lookup_dflt_path_fn
        calib_info = lookup_pulse_row_in_csv(path_fn, pulse)
    if isinstance(calib_info, Exception):
        raise calib_info
    try:
        calcam_calib = calib_info['calcam_calibration_file']
    except KeyError as e:
        raise ValueError(f'Calcam calib lookup file does not contain column "calcam_calibration_file": {path_fn}')
    return calcam_calib

def lookup_pulse_row_in_csv(path_fn: Union[str, Path], pulse: int) -> Union[pd.Series, Exception]:
    """Return row from csv file containing information for pulse range containing supplied pulse number

    :param path_fn: path to csv file containing pulse range information
    :param pulse: pulse number of interest
    :return: Pandas Series containing pulse information / Exception if unsuccessful
    """
    try:
        table = pd.read_csv(path_fn)
    except FileNotFoundError:
        return FileNotFoundError(f'{path_fn}')
    assert (np.all(col in list(table.columns) for col in ['pulse_start', 'pulse_end']),
            'CSV file must contain "pulse_start", "pulse_end" columns')
    row_mask = np.logical_and(table['pulse_start'] <= pulse, table['pulse_end'] >= pulse)
    print(row_mask)
    if np.sum(row_mask) > 1:
        raise ValueError(f'Calcam calib lookup file contains overlapping ranges. Please fix: {path_fn}')
    elif np.sum(row_mask) == 0:
        pulse_ranges = list(zip(table['pulse_start'], table['pulse_end']))
        return ValueError(f'Pulse {pulse} does not fall in any pulse range {pulse_ranges} in {path_fn}')

    return table.loc[row_mask].iloc[0]

if __name__ == '__main__':
    pass