# -*- coding: future_fstrings -*-
#!/usr/bin/env python

"""


Created: 
"""

import logging, time
from typing import Union, Iterable, Tuple, Optional
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import stats

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def identify_bad_frames(frame_data, bit_depth=None, tol_discontinuities=0.01,
                        raise_on_saturated=False, raise_on_uniform=False, raise_on_sudden_intensity_changes=False):
    t0 = time.time()

    bad_frame_info = {}

    # TODO: Find frames with nan values and zeros like last frame of AIT 27880
    # TODO: Find frames with bad time stamps eg t=0 for last frame of AIT 27880

    uniform_frames = identify_uniform_frames(frame_data, raise_on_uniform=raise_on_uniform)
    bad_frame_info['uniform'] = uniform_frames

    discontinuous_frames = identify_sudden_intensity_changes(frame_data, tol=tol_discontinuities,
                                                             raise_on_sudden_intensity_changes=raise_on_sudden_intensity_changes)
    bad_frame_info['discontinuous'] = discontinuous_frames

    if bit_depth is not None:
        saturated_frames = identify_saturated_frames(frame_data, bit_depth,
                                                     raise_on_saturated=False)
        bad_frame_info['saturated'] = saturated_frames
    else:
        logger.warning(f'Cannot check for saturated pixels due to unknown camera bit depth')

    t1 = time.time()
    logger.debug(f'Assessed {len(frame_data)} frame movie for bad frames in {t1-t0:0.3g}s')

    # TODO: Add function to pick out frames that are bad enough to act on - eg opening/closing bad frames

    return bad_frame_info

def identify_uniform_frames(frame_data, raise_on_uniform=True):
    assert np.array(frame_data).ndim == 3

    if isinstance(frame_data, xr.DataArray):
        frame_nos = frame_data['n'].values
    else:
        frame_nos = np.arange(len(frame_data))

    uniform_frames = []
    uniform_frame_values = []
    for n, frame in zip(frame_nos, frame_data):
        frame = np.array(frame).ravel()
        if np.all(frame == frame[0]):
            uniform_frames.append(n)
            uniform_frame_values.append(frame[0])
        elif np.all(np.isnan(frame)):
            uniform_frames.append(n)
            uniform_frame_values.append(frame[0])

    n_bad = len(uniform_frames)
    if n_bad > 0:
        message = (f'Movie data contains uniform/dropped frames for '
                   f'{n_bad}/{len(frame_data)} frames. '
                   f'Frames: {uniform_frames} '
                   f'Values: {uniform_frame_values}')
        if raise_on_uniform:
            raise ValueError(message)
        else:
            logger.warning(message)

    out = dict(frames=uniform_frames, uniform_frame_values=uniform_frame_values, n_bad_frames=n_bad)

    return out


def identify_saturated_frames(frame_data: xr.DataArray, bit_depth, raise_on_saturated=True):
    # TODO: Return number and pix coords of saturated pixels
    saturation_dl = 2**int(bit_depth) - 1

    hyper_saturated_frames = frame_data.where(frame_data > saturation_dl, drop=True).coords
    if len(hyper_saturated_frames['n']) > 0:
        raise ValueError(f'Frame data contains intensities above saturation level for bit_depth={bit_depth}:\n'
                         f'{hyper_saturated_frames}')

    saturated_frames = frame_data.where(frame_data == saturation_dl, drop=True).coords

    n_bad_pixels = len(saturated_frames)
    n_bad_frames = len(saturated_frames['n'])

    if len(saturated_frames['n']) > 0:
        message = (f'Movie data contains {n_bad_pixels} saturated pixels (DL=2^{bit_depth}={saturation_dl}) across '
                   f'{n_bad_frames}/{len(frame_data)} frames:\n)'
                   f'{saturated_frames}')
        if raise_on_saturated:
            raise ValueError(message)
        else:
            logger.warning(message)

    out = dict(frames=saturated_frames, saturation_dl=saturation_dl, bit_depth=bit_depth, n_bad_pixels=n_bad_pixels,
               n_bad_frames=n_bad_frames)

    return out

def identify_sudden_intensity_changes(frame_data: xr.DataArray, tol: float=0.01,
                                      raise_on_sudden_intensity_changes: bool=True):
    """Useful for identifying dropped for faulty frames"""

    frame_intensities = frame_data.sum(dim=['x_pix', 'y_pix'])
    diffs = np.abs(frame_intensities.diff(dim='n'))
    # discontinuous_frames = diffs.where(diffs > (diffs.mean() * tol), drop=True).coords
    # TODO: Use more robust method of identifying outliers?
    nsigma = calc_outlier_nsigma_for_sample_size(len(frame_data), tol=tol)
    discontinuous_frames = diffs.where(diffs > (diffs.mean() + diffs.std() * nsigma), drop=True).coords

    n_bad = len(discontinuous_frames['n'])
    if n_bad > 0:
        message = (f'Movie data contains sudden discontinuities in intensity '
                   f'for {n_bad}/{len(frame_data)} frames (diff>mu+{nsigma:0.3f}*sigma):\n)'
                   f'{discontinuous_frames}')
        if raise_on_sudden_intensity_changes:
            diffs.plot()
            raise ValueError(message)
        else:
            logger.warning(message)

    out = dict(frames=discontinuous_frames, n_bad_frames=n_bad, nsigma=tol)

    return out

def calc_outlier_nsigma_for_sample_size(n_data, tol=0.01):
    """Return number of standard deviations that has a (1-tol) probability of containing all values (assuming a
    normal distribution). Useful for identifying thresholds used to identify outliers.

    The more samples you have the more likely you are to sample the extrema of your distribution. Eg if your sample
    size is 1e10 you can expect 2.7e7 values to be outside 3 sigma. Therefore this function can tell you how many
    sigma to go to for it to be unlikely to return more extreme values. In this case, to have a 1% chance of finding
    a more extreme point you need to go to 7.1 sigma

    eg for data with 100 samples the number of standard deviations you need to go to to expect to find 5 values
    outside nsigma is nsigma = calc_outlier_nsigma_for_sample_size(100, tol=0.01) = 1.95 sigma ~ 2sigma
    ie the normal rule of thumb of ~5% of values falling outside 2 sigma, ~0.3% of values outside 3 sigma

    Args:
        n_data: Sample size
        tol: Number of points you want to (on average) fall outside the returned nsigna

    Returns: number of standard deviations

    """
    # Note: factor of 2 is to account for positive and negative tails of distribution
    nsigma = -stats.norm.ppf(tol/(2*n_data))
    return nsigma

def remove_bad_opening_and_closing_frames(frame_data, bad_frames):
    info = {'start': 0, 'end': 0, 'n_removed': []}


    n = len(frame_data)-1
    i = 0

    while i in bad_frames:
        frame_data = frame_data[1:]
        i -= 1
        info['start'] += 1
        info["n_removed"].append(i)

    while n in bad_frames:
        frame_data = frame_data[:-1]
        n -= 1
        info['end'] += 1
        info["n_removed"].append(n)

    if info['start'] > 0:
        logger.warning(f'Removed {info["start"]} bad frames from start of movie: {info["n_removed"]}')
    if info['end'] > 0:
        logger.warning(f'Removed {info["end"]} bad frames from end of movie: {info["n_removed"]}')

    return frame_data, info


if __name__ == '__main__':
    pass