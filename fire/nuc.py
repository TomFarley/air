# -*- coding: future_fstrings -*-
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

def get_nuc_frame(origin: Union[Dict, str, Path]='first_frame', frame_data: Optional[xr.DataArray]=None,
                  reduce_func: str='mean') -> xr.DataArray:
    if origin in ('first_frame', 'first frame'):
        origin = {'n': [0, 0]}
    elif isinstance(origin, (str, Path)):
        load_nuc_frame_from_file(origin)
    else:
        assert isinstance(origin, dict) and len(origin) == 1, f'Origin dict must have format {{coord: [<coor_range>]}}'
        assert isinstance(frame_data, xr.DataArray), (f'Need DataArray frame_data from which to index NUC frame: '
                                                      f'frame_data={frame_data}, origin={origin}')
    coord, coord_range = list(origin.items())[0]
    coord_slice = slice(coord_range[0], coord_range[1])
    nuc_frame = frame_data.sel({coord: coord_slice})
    nuc_frame = getattr(nuc_frame, reduce_func)(dim='n')

    # TODO: Check properties of frame are appropriate for NUC ie. sufficiently uniform and dim
    return nuc_frame

def load_nuc_frame_from_file(path_fn: Union[Path, str]):
    if not Path(path_fn).exists():
        raise FileNotFoundError(f'Supplied NUC file path does not exist: "{path_fn}"')
    raise NotImplementedError

def apply_nuc_correction(frame_data: xr.DataArray, nuc_frame: xr.DataArray, raise_on_negatives: bool=True):
    frame_data = frame_data - nuc_frame
    if np.any(frame_data < 0):
        frames_with_negatives = frame_data.where(frame_data < 0, drop=True).coords
        message = (f'NUC corrected frame data contains negative intensities for '
                   f'{len(frames_with_negatives["n"])}/{len(frame_data)} frame numbers:\n{frames_with_negatives}\n'
                   f'Setting negative values to zero.')
        if raise_on_negatives:
            raise ValueError(message)
        else:
            logger.warning(message)
            frame_data = xr.apply_ufunc(np.clip, frame_data, 0, None)
    # TODO: Check for negative values etc
    assert not np.any(frame_data < 0), f'Negative values have not been clipped after NUC'
    return frame_data

if __name__ == '__main__':
    pass