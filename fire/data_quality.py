# -*- coding: future_fstrings -*-
#!/usr/bin/env python

"""


Created: 
"""

import logging
from typing import Union, Iterable, Tuple, Optional
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def identify_saturated_frames(frame_data: xr.DataArray, bit_depth, raise_on_saturated=True):
    saturation_dl = 2**bit_depth - 1

    hyper_saturated_frames = frame_data.where(frame_data > saturation_dl, drop=True).coords
    if len(hyper_saturated_frames['n']) > 0:
        raise ValueError(f'Frame data contains intensities above saturation level for bit_depth={bit_depth}:\n'
                         f'{hyper_saturated_frames}')

    saturated_frames = frame_data.where(frame_data == saturation_dl, drop=True).coords
    if len(saturated_frames['n']) > 0:
        message = (f'Movie data contains saturated pixels (DL=2^{bit_depth}={saturation_dl}) for '
                   f'{len(saturated_frames["n"])}/{len(frame_data)} frames:\n)'
                   f'{saturated_frames}')
        if raise_on_saturated:
            raise ValueError(message)
        else:
            logger.warning(message)
    return saturated_frames

if __name__ == '__main__':
    pass