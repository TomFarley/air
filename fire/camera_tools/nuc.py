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
# logger.setLevel(logging.DEBUG)

def get_nuc_frames(origin: Union[Dict, str, Path]= 'first_frame', frame_data: Optional[xr.DataArray]=None,
                   reduce_func: str='mean', n_digitisers=1) -> xr.DataArray:
    nuc_frames = nuc_frames = xr.DataArray(coords=dict(y_pix=frame_data['y_pix'], x_pix=frame_data['x_pix'],
                                    i_digitiser=np.arange(n_digitisers)), dims=('i_digitiser', 'y_pix',
                                                                                'x_pix')).astype(np.uint16)
    nuc_info = nuc_frames.attrs
    nuc_info['digitiser_frames'] = dict()
    nuc_info['digitiser'] = np.zeros_like(frame_data['n'])

    for i_digitiser in np.arange(n_digitisers):
        frame_data_i = frame_data[i_digitiser::n_digitisers]
        frame_nos_i = frame_data_i['n'].values
        if isinstance(origin, (str, Path)):
            nuc_frame = load_nuc_frame_from_file(origin)
        else:
            if origin in ('first_frame', 'first frame'):
                n0 = frame_nos_i[0]
                origin = {'n': [n0, n0]}
            else:
                assert isinstance(origin, dict) and len(origin) == 1, (f'Origin dict must have format '
                                                                       f'{{coord: [<coord_range>]}}')
                assert isinstance(frame_data, xr.DataArray), (f'Need DataArray frame_data from which to index NUC frame: '
                                                              f'frame_data={frame_data}, origin={origin}')
            coord, coord_range = list(origin.items())[0]
            coord_slice = slice(coord_range[0], coord_range[1])
            nuc_frame = frame_data_i.sel({coord: coord_slice})
            nuc_frame = getattr(nuc_frame, reduce_func)(dim='n', skipna=True)
            nuc_frame = nuc_frame.astype(int)  # mean etc can give float output
            nuc_frame = nuc_frame.expand_dims(dim=dict(i_digitiser=[i_digitiser]))
            nuc_frame.name = f'nuc_frame_{i_digitiser}'

            nuc_info['digitiser_frames'][i_digitiser] = frame_nos_i
            nuc_info['digitiser'][i_digitiser::n_digitisers] = i_digitiser
        nuc_frames.loc[nuc_frame.coords] = nuc_frame
    nuc_info['origin'] = origin
    nuc_info['n_digitisers'] = n_digitisers

    # TODO: Check properties of frame are appropriate for NUC ie. sufficiently uniform and dim
    return nuc_frames

def load_nuc_frame_from_file(path_fn: Union[Path, str]):
    if not Path(path_fn).exists():
        raise FileNotFoundError(f'Supplied NUC file path does not exist: "{path_fn}"')
    raise NotImplementedError
    return nuc_frame

def apply_nuc_correction(frame_data: xr.DataArray, nuc_frames: xr.DataArray, raise_on_negatives: bool=True):
    frame_data = frame_data.astype(np.int32)  # Make signed int so can have negative values after subtraction
    for i_digitiser in nuc_frames['i_digitiser']:
        digitiser_frames = nuc_frames.attrs['digitiser_frames'][int(i_digitiser)]
        frame_data.loc[dict(n=digitiser_frames)] -= nuc_frames.loc[i_digitiser].astype(np.int32)
    # frame_data = frame_data - nuc_frames
    if np.any(frame_data < 0):
        frames_with_negatives = frame_data.where(frame_data < 0, drop=True).coords
        message_1 = (f'NUC corrected frame data contains negative intensities for '
                   f'{len(frames_with_negatives["n"])}/{len(frame_data)}. Setting negative values to zero. ')
        message_2 = (f'NUC corrected frames with negative intensities:\n{frames_with_negatives}')
        if raise_on_negatives:
            raise ValueError(message_1 + message_2)
        else:
            logger.warning(message_1)
            logger.debug(message_2)
            frame_data = xr.apply_ufunc(np.clip, frame_data, 0, None)
    # TODO: Check for negative values etc
    assert not np.any(frame_data < 0), f'Negative values have not been clipped after NUC'
    return frame_data

if __name__ == '__main__':
    pass