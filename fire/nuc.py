#!/usr/bin/env python

"""


Created: 
"""

import logging
from typing import Union, Iterable, Tuple, Optional, Dict
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def get_nuc_frame(frame_data: xr.DataArray, origin: Union[Dict, str, Path]='first_frame') -> xr.DataArray:
    if origin == 'first_frame':
        origin = {'n': [0, 1]}
    elif isinstance(origin, (str, Path)):
        assert Path(origin).exists(), f'Supplied NUC file path does not exist: {origin}'
        raise NotImplementedError
    else:
        assert isinstance(origin, dict) and len(origin) == 1, f'Origin dict must have format {{coord: [<coor_range>]}}'
    coord, coord_range = list(origin.items())[0]
    coord_slice = slice(coord_range[0], coord_range[1])
    nuc_frame = frame_data.sel({coord: coord_slice}, method='nearest').mean('n-')

    # TODO: Check properties of frame are appropriate for NUC ie. sufficiently uniform and dim
    return nuc_frame

if __name__ == '__main__':
    pass