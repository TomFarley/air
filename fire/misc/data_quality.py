# -*- coding: future_fstrings -*-
#!/usr/bin/env python

"""


Created: 
"""

import logging, time
from typing import Union, Iterable, Tuple, Optional
from pathlib import Path
from copy import copy

import numpy as np
import xarray as xr
from scipy import stats, interpolate
import matplotlib.pyplot as plt

from fire.misc import data_structures
from fire.plotting import plot_tools

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

info_removed_frames = {'n_corrected': 0, 'corrected': [],
                       'n_removed_start': 0, 'n_removed_end': 0, 'n_removed': 0,
                       'removed_start': [], 'removed_end': [], 'removed': [],
                        'n_interpolated_middle': 0, 'n_naned_middle': 0,
                       'interpolated_middle': [], 'naned_middle': []}

def identify_bad_frames(frame_data, bit_depth=None, n_discontinuities_expected=0.01, n_sigma_multiplier=1,
                        debug_plot=False, raise_on_saturated=False, raise_on_uniform=False,
                        raise_on_sudden_intensity_changes=False):
    t0 = time.time()

    bad_frame_info = {}

    # TODO: Find frames with nan values and zeros like last frame of AIT 27880
    # TODO: Find frames with bad time stamps eg t=0 for last frame of AIT 27880

    uniform_frames = identify_uniform_frames(frame_data, raise_on_uniform=raise_on_uniform)
    bad_frame_info['uniform'] = uniform_frames

    discontinuous_frames = identify_sudden_intensity_changes(frame_data, n_outliers_expected=n_discontinuities_expected,
                                                             n_sigma_multiplier=n_sigma_multiplier,
                            raise_on_sudden_intensity_changes=raise_on_sudden_intensity_changes, debug_plot=debug_plot)
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

def identify_sudden_intensity_changes(frame_data: xr.DataArray, n_outliers_expected: float=1, n_sigma_multiplier=1,
                                      raise_on_sudden_intensity_changes: bool=True, debug_plot: bool=True):
    """Useful for identifying dropped or faulty frames"""

    frame_data = data_structures.swap_xarray_dim(frame_data, 'n')

    frame_intensities = frame_data.astype(np.int64).sum(dim=['x_pix', 'y_pix'])
    diffs = frame_intensities.diff(dim='n')
    diffs_abs = np.abs(diffs)
    # If there are several bad frames in a row they will be followed by an opposite (normally negative) diff that
    # will bring the cumsum back close to zero
    diffs_cum_sum = diffs.cumsum(dim='n')
    diffs_cum_sum_abs = np.abs(diffs_cum_sum)
    # diffs['n'] -= 1
    # diffs['t'] -= (diffs['t'][1] - diffs['t'][0])
    # discontinuous_frames = diffs.where(diffs > (diffs.mean() * tol), drop=True).coords
    # TODO: Use more robust method of identifying outliers?
    nsigma = calc_outlier_nsigma_for_sample_size(len(frame_data), n_outliers_expected=n_outliers_expected)

    # Raw diff is useful for catching bad starting frames
    diff_threshold = (diffs_abs.mean() + diffs_abs.std() * (nsigma*n_sigma_multiplier))
    diff_mask = diffs_abs > diff_threshold
    # TODO: Avoid raw diff marking negative diff following positive diff (ie first subsequent good frame) as bad

    # Cumulative diff is useful for finding sets of bad frames in a row
    diff_sum_abs_threshold = (diffs_cum_sum_abs.mean() + diffs_cum_sum_abs.std() * (nsigma*n_sigma_multiplier))
    diff_cum_abs_mask = diffs_cum_sum_abs > diff_sum_abs_threshold

    # Combine masks for start and middle
    discontinuous_mask = diff_mask + diff_cum_abs_mask

    # Generally want to identify a frame if it is a sudden change from the previous frame, but in case of first
    # frame want to identify first frame as discontinuous if it differs a lot from following frame
    # Diff are indexed starting at 1 whereas we want to get a full length mask starting at 0
    n_discontinuous_start = 0
    while discontinuous_mask[n_discontinuous_start]:
        n_discontinuous_start += 1
    # n = discontinuous_mask['n'].values
    # t = discontinuous_mask['t'].values
    # mask_values = discontinuous_mask.values
    if n_discontinuous_start > 0:
        dt = float(diffs['t'][1] - diffs['t'][0])

        n_dim_modified = copy(discontinuous_mask['n'].values)
        n_dim_modified[:n_discontinuous_start] -= 1
        # Need to make sure coord is active before it can be modified
        discontinuous_mask = data_structures.swap_xarray_dim(discontinuous_mask, 'n')
        diffs = data_structures.swap_xarray_dim(diffs, 'n')
        discontinuous_mask['n'] = n_dim_modified
        diffs['n'] = n_dim_modified

        t_dim_modified = copy(discontinuous_mask['t'].values)
        t_dim_modified[:n_discontinuous_start] -= dt
        # Need to make sure coord is active before it can be modified
        discontinuous_mask = data_structures.swap_xarray_dim(discontinuous_mask, 't')
        diffs = data_structures.swap_xarray_dim(diffs, 't')
        discontinuous_mask['t'] = t_dim_modified
        diffs['t'] = t_dim_modified

        coords = {'n': [discontinuous_mask['n'][n_discontinuous_start - 1] + 1],
                  't': ('n', [discontinuous_mask['t'][n_discontinuous_start - 1] + dt])}

        gap_value = xr.DataArray([False], dims=['n'], coords=coords)
        discontinuous_mask = data_structures.swap_xarray_dim(discontinuous_mask, 'n')
        discontinuous_mask = xr.concat([discontinuous_mask, gap_value], dim='n').sortby('n')

        gap_value = xr.DataArray([0], dims=['n'], coords=coords)
        diffs = data_structures.swap_xarray_dim(diffs, 'n')
        diffs = xr.concat([diffs, gap_value], dim='n').sortby('n')

    discontinuous_frames = diffs.where(discontinuous_mask, drop=True).coords

    if debug_plot:
        fig, ax, ax_passed = plot_tools.get_fig_ax()
        frame_intensities.plot(ax=ax, label='Frame summed intensity', ls=':')
        diffs.plot(ax=ax, label='Diffs', ls=':')
        diffs_abs.plot(ax=ax, label='Diffs abs', ls='--')
        diffs_cum_sum_abs.plot(ax=ax, label='Diffs cum sum abs')
        ax.axhline(y=diff_threshold, ls='--', color='k', label='Diffs abs threshold')
        ax.axhline(y=diff_sum_abs_threshold, ls='-', color='k', label='Diffs cum sum threshold')
        for n in discontinuous_frames['n']:
            ax.axvline(x=n, ls=':', color='k', lw=1, alpha=0.7)
        plot_tools.legend(ax)
        plot_tools.show_if(True, tight_layout=True)

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

    out = dict(frames=discontinuous_frames, n_bad_frames=n_bad, nsigma=nsigma, n_outliers_expected=n_outliers_expected)

    return out

def calc_outlier_nsigma_for_sample_size(n_data, n_outliers_expected=1):
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
        n_data:              Sample size
        n_outliers_expected: Number of points you want to (on average) fall outside the returned nsigna

    Returns: number of standard deviations to have n_outliers_expected number of outlier points outside nsigma

    """
    outlier_fraction = n_outliers_expected / n_data
    # Note: factor of 2 is to account for positive and negative tails of distribution
    qualtile_two_tails = outlier_fraction / 2

    nsigma = -stats.norm.ppf(qualtile_two_tails)

    return nsigma

def remove_bad_frames(frame_data, bad_frames, remove_opening_closing=True, interpolate_middle=True, nan_middle=False,
                                                                                        debug_plot=False):
    info = copy(info_removed_frames)

    def plot_bad_frame(n, debug):
        if debug:
            plt.figure(num=f'Bad frame {int(n)}')
            plt.imshow(frame_data[n])
            plt.show()

    if remove_opening_closing:
        i = 0
        while i in bad_frames:
            info['n_corrected'] += 1
            info['corrected'].append(i)
            info['n_removed_start'] += 1
            info["removed_start"].append(i)
            info["removed"].append(i)
            plot_bad_frame(i, debug_plot)
            i += 1

        n = len(frame_data) - 1
        while n in np.array(bad_frames):
            info['n_corrected'] += 1
            info['corrected'].append(n)
            info['n_removed_end'] += 1
            info["removed_end"].append(n)
            info["removed"].append(n)
            plot_bad_frame(n, debug_plot)
            n -= 1

    if interpolate_middle:
        # Occasionally there are bad frames in the middle of the movie. To prevent these messing up heat flux
        # calculations, while preserving time axis spacing, interpolate these bad frames from good neighbours
        for n in np.array(bad_frames):
            if n not in info['corrected']:
                info['n_corrected'] += 1
                info['corrected'].append(n)
                info['n_interpolated_middle'] += 1
                info["interpolated_middle"].append(n)

        if info['n_interpolated_middle'] > 0:
            mask_bad = frame_data['n'].isin(info["corrected"])
            frames_ok = frame_data.where(~mask_bad, drop=True)
            f = interpolate.interp1d(frames_ok['n'].values, frames_ok.values, kind='linear', axis=0)
            frame_data.loc[mask_bad] = np.round(f(frame_data['n'].where(mask_bad, drop=True)))
    elif nan_middle:
        for n in bad_frames:
            if n not in info['corrected']:
                plot_bad_frame(n, debug_plot)
                frame_data[n] = np.nan  # Setting uint16 to nan will actually set it to zero
                info['n_corrected'] += 1
                info['corrected'].append(i)
                info['n_naned_middle'] += 1
                info["naned_middle"].append(i)

    frame_data = frame_data[info["n_removed_start"]:]
    frame_data = frame_data[:-(info["n_removed_end"]+1)]

    if info['n_removed_start'] > 0:
        logger.warning(f'{info["n_removed_start"]} bad frames at start of movie: {info["removed_start"]}')
    if info['n_removed_end'] > 0:
        logger.warning(f'{info["n_removed_end"]} bad frames at end of movie: {info["removed_end"]}')
    if info['n_interpolated_middle'] > 0:
        logger.warning(f'Interpolated {info["n_interpolated_middle"]} bad frames in middle of movie based on good '
                       f'neighbours: {info["interpolated_middle"]}')
    if info['n_naned_middle'] > 0:
        logger.warning(f'Set {info["n_naned_middle"]} bad frames in middle of movie movie to nans: '
                       f'{info["naned_middle"]}')

    return frame_data, info



    return frame_data, info

if __name__ == '__main__':
    pass