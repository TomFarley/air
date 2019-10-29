#!/usr/bin/env python

"""Functions for interfacing with Calcam


Created: 11-10-19
"""

import logging
from typing import Union, Iterable, Tuple, Optional
from pathlib import Path

import numpy as np

import calcam

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

calcam_calib_dir = Path('/home/tfarley/calcam2/calibrations/')

def get_calcam_calib_path_fn(shot: int, camera: str, machine: str):
    """Return path to calcam calibration file for given discharge and camera

    :param shot: Shot/pulse number
    :param camera: Camera to look up calbration for
    :param machine: Tokamak under analysis
    :return: calib_path_fn
    """
    if (shot == 23586) and (camera=='rit') and (machine=='MAST'):
        fn = 'MAST-rit-p23586-n217-enhanced_1-rough_test.ccc'
        calib_path_fn = calcam_calib_dir / fn
    else:
        raise NotImplementedError

    return calib_path_fn

if __name__ == '__main__':
    pass