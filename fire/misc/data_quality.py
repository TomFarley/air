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

from fire.physics import physics_parameters
from fire.misc import data_structures
from fire.plotting import plot_tools

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

info_removed_frames = {'n_corrected': 0, 'corrected': [],
                       'n_removed_start': 0, 'n_removed_end': 0, 'n_removed': 0,
                       'removed_start': [], 'removed_end': [], 'removed': [],
                        'n_interpolated_middle': 0, 'n_naned_middle': 0,
                       'interpolated_middle': [], 'naned_middle': []}

def identify_bad_frames(frame_data, bit_depth=None, n_discontinuities_expected=0.01,
                        n_sigma_multiplier_high=1, n_sigma_multiplier_low=1, n_sigma_multiplier_start=1,
                        debug_plot=2, scheduler=False, meta_data=None, raise_on_saturated=False,
                        raise_on_uniform=False, raise_on_sudden_intensity_changes=False):
    t0 = time.time()

    bad_frame_info = {}

    # TODO: Find frames with nan values and zeros like last frame of AIT 27880
    # TODO: Find frames with bad time stamps eg t=0 for last frame of AIT 27880

    uniform_frames = identify_uniform_frames(frame_data, raise_on_uniform=raise_on_uniform)
    bad_frame_info['uniform'] = uniform_frames

    discontinuous_frames = identify_sudden_intensity_changes(frame_data, n_outliers_expected=n_discontinuities_expected,
                             n_sigma_multiplier_high=n_sigma_multiplier_high,
                             n_sigma_multiplier_low=n_sigma_multiplier_low,
                             n_sigma_multiplier_start=n_sigma_multiplier_start,
                             meta_data=meta_data, scheduler=scheduler,
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

def shift_starting_diff_coords(discontinuous_mask):
    """Generally want to identify a frame if it is a sudden change from the previous frame, but in case of first
    frame want to identify first frame as discontinuous if it differs a lot from following frame
    Diff are indexed starting at 1 whereas we want to get a full length mask starting at 0"""
    n_discontinuous_start = 0
    while discontinuous_mask[n_discontinuous_start]:
        n_discontinuous_start += 1

    if n_discontinuous_start > 0:
        n = discontinuous_mask['n']
        t = discontinuous_mask['t']
        dt = float(t[2] - t[1])

        n_dim_modified = copy(n.values)
        n_dim_modified[:n_discontinuous_start] -= 1
        # Need to make sure coord is active before it can be modified
        discontinuous_mask = data_structures.swap_xarray_dim(discontinuous_mask, 'n')
        discontinuous_mask['n'] = n_dim_modified

        t_dim_modified = copy(t.values)
        t_dim_modified[:n_discontinuous_start] -= dt
        # Need to make sure coord is active before it can be modified
        discontinuous_mask = data_structures.swap_xarray_dim(discontinuous_mask, 't')
        discontinuous_mask['t'] = t_dim_modified

        coords_insert = {'n': [discontinuous_mask['n'][n_discontinuous_start - 1] + 1],
                         't': ('n', [discontinuous_mask['t'][n_discontinuous_start - 1] + dt])}

        gap_value = xr.DataArray([False], dims=['n'], coords=coords_insert)
        discontinuous_mask = data_structures.swap_xarray_dim(discontinuous_mask, 'n')
        discontinuous_mask = xr.concat([discontinuous_mask, gap_value], dim='n').sortby('n')

    return discontinuous_mask, n_discontinuous_start

def identify_sudden_intensity_changes(frame_data: xr.DataArray, n_outliers_expected: float=1,
                                      n_sigma_multiplier_high=1, n_sigma_multiplier_low=1, n_sigma_multiplier_start=1,
                                      raise_on_sudden_intensity_changes: bool=True, meta_data=None,
                                      debug_plot=2, scheduler=False):
    """Useful for identifying dropped or faulty frames"""

    frame_data = data_structures.swap_xarray_dim(frame_data, 'n').astype(np.int32)  # Prevent sum giving overflow

    nsigma = calc_outlier_nsigma_for_sample_size(len(frame_data), n_outliers_expected=n_outliers_expected)

    # stats = ('min', ('percentile', 2), 'mean', ('percentile', 50), 'std', ('percentile', 98), 'max', 'ptp', 'sum')
    stats = (
        ('percentile', 2),
        # 'std', 'ptp', 'sum'
    )

    stats, labels = physics_parameters.calc_2d_profile_param_stats(frame_data, coords_reduce=('x_pix', 'y_pix'),
                    roll_width=None, stats=stats, kwargs=dict(skipna=True))
    stats_diffs = xr.Dataset()

    for stat in list(stats.keys()):
        value = stats[stat]
        diffs = value.diff(dim='n')
        diffs_abs = np.abs(diffs)
        diffs_cumsum = diffs.cumsum(dim='n')
        diffs_cumsum_abs = np.abs(diffs_cumsum)

        tol_diffs_abs = (diffs_abs.mean() + diffs_abs.std() * (nsigma * n_sigma_multiplier_high))
        tol_diffs_cumsum_high = (diffs_cumsum.mean() + diffs_cumsum.std() * (nsigma * n_sigma_multiplier_high))
        tol_diffs_cumsum_low = (diffs_cumsum.mean() - diffs_cumsum.std() * (nsigma * n_sigma_multiplier_low))

        mask_diffs_abs = (diffs_abs >= tol_diffs_abs)
        mask_diffs_cumsum_high = (diffs_cumsum >= tol_diffs_cumsum_high)
        mask_diffs_cumsum_low = (diffs_cumsum <= tol_diffs_cumsum_low)
        mask_diffs_cumsum = mask_diffs_cumsum_high + mask_diffs_cumsum_low
        mask_diffs_cumsum.name = 'mask_diffs_cumsum'

        tol_first_diff = diffs_cumsum.std() * (nsigma * n_sigma_multiplier_start)
        if np.abs(diffs_cumsum[0]) >= tol_first_diff:
            # cumsum doesn't catch large initial diff
            mask_diffs_cumsum[0] = True

        # Cut out further starting frames if they have even small intensity drops (which lead to -ve heat flux at start)
        for diff in diffs:
            if diff < 0:
                mask_diffs_cumsum.loc[diff['n']] = True
            else:
                break

        n_diffs_abs = mask_diffs_abs['n'].where(mask_diffs_abs, drop=True)
        n_diffs_cumsum = mask_diffs_cumsum['n'].where(mask_diffs_cumsum, drop=True)

        stats_diffs[f'{stat}_diff'] = diffs
        stats_diffs[f'{stat}_diff_cumsum'] = diffs_cumsum

        stats_diffs[f'{stat}_diff_abs_tol'] = tol_diffs_abs
        stats_diffs[f'{stat}_diff_cumsum_tol_low'] = tol_diffs_cumsum_low
        stats_diffs[f'{stat}_diff_cumsum_tol_high'] = tol_diffs_cumsum_high

        stats_diffs[f'{stat}_diff_abs_mask'] = mask_diffs_abs
        stats_diffs[f'{stat}_diff_cumsum_mask'] = mask_diffs_cumsum

        stats_diffs[f'{stat}_diff_abs_frames'] = n_diffs_abs
        stats_diffs[f'{stat}_diffs_cumsum_frames'] = n_diffs_abs

        stats_diffs[f'{stat}_diff_abs_nframes'] = len(n_diffs_abs)
        stats_diffs[f'{stat}_diffs_cumsum_nframes'] = len(n_diffs_cumsum)

        if False:
            fig, ax = plt.subplots(num=stat)
            ax.plot(diffs_cumsum, alpha=0.7, label=f'diffs_cumsum "{stat}"')
            ax.plot(diffs_abs, ls='--', alpha=0.7, label=f'diffs_abs "{stat}"'),
            ax.axhline(tol_diffs_cumsum_low, c='k')
            ax.axhline(tol_diffs_cumsum_high, c='k')
            ax.axhline(tol_diffs_abs, ls='--', c='k')
            plot_tools.legend(ax)
            plot_tools.show_if(True)

    # Need to combine metrics:
    # drop in 2% diff cumsum is good for detecting bad frames in middle of movie
    # drop in std diff cumsum is good for detecting bad frames in middle of movie?
    stats_combine = ['frame_data_2percentile(n)_diff_cumsum',
                     # 'frame_data_2percentile(n)_diff_abs_mask', # Avoid abs mask as poor for consecutive bad frames
                     # 'frame_data_ptp(n)_diff_cumsum_mask',
                     # 'frame_data_std(n)_diff_cumsum'
                     ]

    discontinuous_mask = None
    for stat in stats_combine:
        stat = stat + '_mask'
        if discontinuous_mask is None:
            # First frame missing from diffs so drop nan in dataset with existing full coords
            discontinuous_mask = stats_diffs[stat].dropna(dim='n')
        else:
            discontinuous_mask = discontinuous_mask + stats_diffs[stat].dropna(dim='n')

    discontinuous_mask, n_discontinuous_start = shift_starting_diff_coords(discontinuous_mask)
    discontinuous_frames = discontinuous_mask['n'].where(discontinuous_mask, drop=True)  # tmp before coord fix
    n_bad = len(discontinuous_frames['n'])

    if (not scheduler) and ((debug_plot is True) or (isinstance(debug_plot, int) and n_bad > debug_plot)):
        fig, ax, ax_passed = plot_tools.get_fig_ax()

        for stat in stats_combine:
            stats_diffs[stat].plot(ax=ax, label=stat, ls='-', alpha=0.6)
            for tol_str in ['_tol', '_tol_low', '_tol_high']:
                key = stat + tol_str
                try:
                    tol = stats_diffs[key]
                except KeyError as e:
                    pass
                else:
                    color = plot_tools.get_previous_line_color(ax)
                    ax.axhline(tol, label=key, color=color, alpha=0.6, lw=1)
        ax.axhline(0, ls=':')

        for n in discontinuous_frames['n']:
            ax.axvline(x=n, ls=':', color='r', lw=1, alpha=0.7)
        plot_tools.annotate_providence(ax, meta_data=meta_data)
        plot_tools.legend(ax, title=f'{n_bad} bad frames: {list(np.array(discontinuous_frames["n"]))}')
        plot_tools.show_if(True, tight_layout=True)

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
                                                                meta_data=None, debug_plot=False, scheduler=False):
    info = copy(info_removed_frames)

    def plot_bad_frame(n, debug, meta_data=meta_data, num='Bad frame {n}'):
        if debug:
            fig, ax = plt.subplots(num=num.format(n=int(n)))
            if meta_data is not None:
                plot_tools.annotate_providence(ax, meta_data=meta_data)
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
            debug = debug_plot or (not scheduler and (info['n_removed_start'] > 1))  # Only plot past first frame
            plot_bad_frame(i, debug, meta_data=meta_data)
            i += 1

        debug = debug_plot or (not scheduler)  # Always plot bad end and middle frames

        n = len(frame_data) - 1
        while n in np.array(bad_frames):
            info['n_corrected'] += 1
            info['corrected'].append(n)
            info['n_removed_end'] += 1
            info["removed_end"].append(n)
            info["removed"].append(n)
            plot_bad_frame(n, debug, meta_data=meta_data)
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
                plot_bad_frame(n, debug, meta_data=meta_data)

        if info['n_interpolated_middle'] > 0:
            mask_bad = frame_data['n'].isin(info["interpolated_middle"])
            frames_ok = frame_data.where(~mask_bad, drop=True)
            func_interp = interpolate.interp1d(frames_ok['n'].values, frames_ok.values, kind='linear', axis=0)
            frames_bad = np.array(frame_data['n'].where(mask_bad, drop=True), dtype=int)
            frame_data.loc[mask_bad] = np.round(func_interp(frames_bad)).astype(int)
            for n in frames_bad:
                plot_bad_frame(n, debug, meta_data=meta_data, num='Interpolated frame {n}')

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

def filter_non_monotonic(dataset, coord, recursive=True, non_monotonic_prev=None):
    dim = dataset[coord].dims[0]

    # Use 'lower' label so that value at start of negative diff is removed rather than at end. Intended effect:
    #  eg keep point on T5 while dropping preceeding point on T4 cutting under T5
    mask_monotonic = dataset[coord].diff(dim=dim, label='lower') > 0  # mask_monotonic.shift(shifts={dim: -1})
    mask_non_monotonic = ~mask_monotonic

    if np.any(mask_non_monotonic):

        data_non_monotonic = dataset.where(mask_non_monotonic, drop=True)
        if non_monotonic_prev is not None:
            data_non_monotonic = xr.concat([data_non_monotonic, non_monotonic_prev], dim=dim).sortby(dim)

        dataset_monotonic = dataset.where(mask_monotonic, drop=True)

        if recursive:
            dataset_monotonic, data_non_monotonic = filter_non_monotonic(dataset_monotonic, coord, recursive=True,
                                                                         non_monotonic_prev=data_non_monotonic)
    else:
        # Break recursion when already monotonic
        dataset_monotonic = dataset
        data_non_monotonic = non_monotonic_prev
        if data_non_monotonic is None:
            data_non_monotonic = xr.Dataset()

    return dataset_monotonic, data_non_monotonic

if __name__ == '__main__':
    pass