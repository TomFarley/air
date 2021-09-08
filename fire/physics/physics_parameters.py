#!/usr/bin/env python

"""


Created: 
"""

import logging, itertools
from collections import defaultdict, namedtuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy import interpolate

import matplotlib.pyplot as plt

from fire.geometry.geometry import (calc_horizontal_path_anulus_areas, calc_tile_tilt_area_correction_factors,
    calc_divertor_area_integrated_param)
from fire.misc.data_structures import attach_standard_meta_attrs, get_reduce_coord_axis_keep, reduce_2d_data_array
from fire.misc.utils import make_iterable
from fire.plotting import debug_plots

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# Sentinel for default keyword arguments
module_defaults = object()

# TODO: Move module defaults to json files
params_dict_default = {'required':
                           ['alpha_const'],
                       'optional':
                            []
                       }

param_funcs = {
    'annulus_areas_horizontal_path': calc_horizontal_path_anulus_areas,
}

legacy_values = {23586:
                     {"AIR_ALPHACONST_ISP": 70000.0,
                     "AIR_ALPHACONST_ISP_ELM": 30000.0,
                     "AIR_CAMERA VIEW_ISP": 2,
                     "AIR_CAMERA VIEW_OSP": 4,
                     "AIR_ERRORFLAG": 110

                      }
                 }


def check_input_params_complete(data, params_dict=module_defaults):
    if params_dict is None:
        params_dict = params_dict_default
    for param in params_dict['required']:
        if param not in data:
            raise KeyError(f'Analysis input parameter "{param}" missing from:\n{data}')

def attach_meta_data(data, meta_data_dict=None):
    raise NotImplementedError

def find_peaks_info(profile, x=None, peak_kwargs=(('width', 3),), add_boundary_local_maxima=False, coords=()):
    from scipy import signal

    if isinstance(peak_kwargs, tuple):
        # NOTE: width arg for find_peaks is measured smaller than expected - not same as order to argrelmax,
        # so best not to specify it...
        peak_kwargs = dict(peak_kwargs)  # (('width', 3),)

    peak_info = defaultdict(list)

    ind_peaks, properties_peaks = signal.find_peaks(profile, **peak_kwargs)
    n_peaks = len(ind_peaks)

    ind_rel_max = signal.argrelmax(np.array(profile), order=peak_kwargs['width'])

    i_max = np.argmax(np.array(profile))
    if i_max in (0, len(profile)):
        # Peak value is at boundaries of data that will not be picked up by scipy's find_peaks
        pass

    if add_boundary_local_maxima:
        # find_peaks can miss local maxima at the edge of the domain due to insufficient width - add in local maxima
        ind_min = np.min(ind_peaks) if (n_peaks > 0) else len(profile)
        for i, ind in enumerate(ind_rel_max):
            if ind < ind_min:
                np.insert(ind_peaks, ind, i)
                properties_peaks = None
                # TODO: insert values for properties also? or set to None to ignore?

        ind_max = np.max(ind_peaks) if (n_peaks > 0) else 0
        for i, ind in enumerate(ind_rel_max[::-1]):
            if ind < ind_max:
                np.insert(ind_peaks, ind, n_peaks-1-i)
                properties_peaks = None
                # TODO: insert values for properties also? or set to None to ignore?


    # find peaks doesn't count nan values towards peak indices, so add them on? NO
    # nan_sum = np.cumsum(np.isnan(profile))
    # for i, i_peak in enumerate(copy(ind_peaks)):
    #     ind_peaks[i] += nan_sum[i_peak]

    amplitude_peaks = profile[ind_peaks]

    # Order peaks in decreasing order
    peak_order = np.argsort(amplitude_peaks)[::-1]
    # ind_peaks = make_iterable(ind_peaks[np.array(peak_order)], cast_to=np.ndarray)
    n_peaks = len(ind_peaks)

    # Global maxima
    if n_peaks > 0:
        i_peak_global = ind_peaks[0]
        amplitude_peak_global = profile[i_peak_global]
    else:
        i_peak_global = np.nan
        amplitude_peak_global = np.nan

    peak_info['amplitude_peaks'] = amplitude_peaks
    peak_info['ind_peaks'] = ind_peaks
    peak_info['order_peaks_by_amplitude'] = peak_order
    peak_info['n_peaks'] = n_peaks
    peak_info['peak_properties'] = properties_peaks

    peak_info['amplitude_peak_global'] = amplitude_peak_global
    peak_info['ind_peak_global'] = i_peak_global

    # If we know the x coordinate, include x coordinates at peaks
    if isinstance(profile, xr.DataArray) and (x is None):
        x = profile[profile.dims[0]]
    if x is not None:
        peak_info['x_peaks'] = x[ind_peaks]
        peak_info['x_peak_global'] = x[i_peak_global] if not np.isnan(i_peak_global) else np.nan

    for name, values in dict(coords).items():
        peak_info[f'{name}_peaks'] = values[ind_peaks]
        peak_info[f'{name}_peak_global'] = values[i_peak_global] if not np.isnan(i_peak_global) else np.nan

    return peak_info


def calc_strike_point_continutity_rating(peaks_info, strike_points, strike_point_index, t,
                                         strike_point_assumptions='continuous',
                                         history_weights=(0.5, 0.2, 0.1, 0.05, 0.05),
                                         scalings=(('displacement', 1.0), ('amplitude', 1.0)),
                                         weightings=(('small_displacement', 1.0), ('similar_amplitude', 0.3),
                                                     ('high_amplitude', 0.2)),
                                         penalties=(('decreasing_R', 0.1), ('stationary', 0.05),
                                                    ('non_monotonic', 0.1),
                                                    ('incomplete_history', 0.1)),
                                         limits=(('max_movement', (-0.1, 0.5)), ('s_fluctuations', 0.005)),
                                         fom_threshold=1e3):
    FomComponents = namedtuple('FomComponents', ['high_amplitude', 'small_displacement', 'similar_amplitude',
                                                 'decreasing_R', 'stationary', 'non_monotonic'])

    scalings = dict(scalings)
    weightings = dict(weightings)
    penalties = dict(penalties)
    limits = dict(limits)

    n_strike_points = len(strike_points.columns.levels[0])
    n_points_history = len(history_weights)
    history_weights = np.array(history_weights)

    i_peak_global = peaks_info[f'ind_global_peak']
    amplitude_peak_global = peaks_info['amplitude_peak_global']

    ind_peaks = peaks_info['ind_peaks']  # indices of peaks
    amplitude_peaks = peaks_info['amplitude_peaks']  # intensities of peaks
    peak_order = peaks_info['order_peaks_by_amplitude']  # indices that order the peak arrays in descending order

    n_peaks = peaks_info['n_peaks']
    properties_peaks = peaks_info['peak_properties']

    s_peaks = np.array(peaks_info['s_global_peaks'])  # s coords of peaks
    r_peaks = np.array(peaks_info['R_peaks'])  # s coords of peaks
    ind_peaks_sorted = ind_peaks[peak_order]

    # TODO: move to function
    # Select data for this strike point index
    strike_point_df_i = strike_points[strike_point_index]

    # Select only non nan values - ie where strike point location found
    mask_not_nan = ~strike_point_df_i['ind'].isnull()
    t_not_nan = np.array(strike_point_df_i.index)[mask_not_nan]
    strike_point_not_nan = strike_point_df_i.loc[t_not_nan]

    n_previous_locations = np.sum(mask_not_nan)

    history = {}
    t_history = t_not_nan[-n_points_history:]
    t_history_plus1 = t_not_nan[-(n_points_history+1):]  # used for diffs
    history['t'] = t_history
    history['inds'] = strike_point_not_nan.loc[t_history, 'ind'].values
    history['R'] = strike_point_not_nan.loc[t_history, 'R'].values
    history['s'] = strike_point_not_nan.loc[t_history, 's_global'].values
    # history['ds'] = np.concatenate([[0]*(n_previous_locations > 0), np.diff(history['s'])])
    history['ds'] = np.concatenate([[0]*(n_previous_locations < (n_points_history+1)),
                                    np.diff(strike_point_not_nan.loc[t_history_plus1, 's_global'].values)])
    history['dR'] = np.concatenate([[0]*(n_previous_locations < (n_points_history+1)),
                                    np.diff(strike_point_not_nan.loc[t_history_plus1, 'R'].values)])
    history['amplitude'] = strike_point_not_nan.loc[t_history, 'amplitude'].values
    history['fom'] = strike_point_not_nan.loc[t_history, 'fom'].values

    # ind_peak_prev = strike_points.loc[coord_last_non_nan, (strike_point_index, 'ind')]
    # s_peak_prev = strike_points.loc[coord_last_non_nan, (strike_point_index, 's_global')]
    # amplitude_peak_prev = strike_points.loc[coord_last_non_nan, (strike_point_index, 'amplitude')]


    # If there are fewer previous points than there are required history, pad values with nans
    n_missing_hsitory = np.max([n_points_history - n_previous_locations, 0])
    if n_missing_hsitory > 0:
        # Pad with nans if not enough points in history for weighted means
        for key in ('t', 'inds', 's', 'R', 'ds', 'dR', 'amplitude', 'fom'):
            history[key] = np.insert(history[key], 0, np.repeat(np.nan, n_missing_hsitory))

    partial_hist_pen = n_missing_hsitory * penalties['incomplete_history']  # Penalise fom using partial hist

    if n_previous_locations > 0:
        # Take weighted average of previous values to inform next selection
        history_nan_mask = ~np.isnan(history['t'])
        # TODO: Look at fom history - use as weighting?
        weights = history_weights[history_nan_mask]  # * history_weights['fom']
        for key in ('t', 'inds', 's', 'R', 'ds', 'dR', 'amplitude'):
            history[f'{key}_av'] = np.average(history[key][history_nan_mask], weights=weights)
            history[f'{key}_std'] = np.std(history[key][history_nan_mask])

        # nans in ds mess things up!
        mask_s_peaks_nan = np.isnan(s_peaks)
        if np.any(mask_s_peaks_nan):
            logger.debug('ds values contain nans - interpolating and extrapolating to fill nan values based on dR')
            print(t)
            if np.sum(~mask_s_peaks_nan) >= 2:
                f = interpolate.InterpolatedUnivariateSpline(r_peaks[~mask_s_peaks_nan], s_peaks[~mask_s_peaks_nan],
                                                             k=1, ext=0)
                s_peaks[mask_s_peaks_nan] = f(r_peaks[mask_s_peaks_nan])
                peaks_info['s_global_peaks'][:] = s_peaks  # Need to update s values outside func
            else:
                pass

        # Displacement from last location to current peak options
        diffs_r = r_peaks - history['R_av']
        diffs_s = s_peaks - history['s_av']
        diffs_amp = amplitude_peaks - history['amplitude_av']
        ds = s_peaks - history['s'][-1]  # Displacement from last location to current peak options
        dr = r_peaks - history['R'][-1]

        # Check if current step is in same direction as prev trend (accepting some noise around 0)
        monotonic = (np.sum(history['ds']) >= -limits['s_fluctuations']) & (ds >= 0)
        stationary = (ds == 0)

        weight_small_dist = weightings['small_displacement'] + partial_hist_pen
        weight_similar_amp = weightings['similar_amplitude'] + partial_hist_pen

        fom_components = FomComponents(
                # High amplitude: Always favor high absolute amplitudes (independent of history)
                # Clip amplitudes at 0 to avoid -ve square roots
                (scalings['amplitude']/np.clip(amplitude_peaks, 1e-3, None)) ** weightings['high_amplitude'],
                # Small displacement: Favor small displacement from last position (less weight with limited history)
                (np.abs(diffs_s)/scalings['displacement']) ** weight_small_dist,
                # Similar_amplitude: Favor similar amplitude to last value (less weight when have limited history)
                (np.abs(diffs_amp) / scalings['amplitude']) ** weight_similar_amp,
                # Apply penalty for strike point moving to lower R (against trend from solenoid sweep)
                (diffs_s < -limits['s_fluctuations']) * penalties['decreasing_R'],
                # Apply penalty for strike point sitting on exactly same spot (avoid getting stuck at tile boundary etc)
                stationary * penalties['stationary'],
                # Non-monotonic: Apply penalty when strike point movement is reversing direction
                ((~monotonic) * penalties['non_monotonic'])
                )
        foms = np.sum(fom_components, axis=0)
        # Established set limits on maximum movement between frames
        outside_limits = (((ds < limits['max_movement'][0]) | (ds > limits['max_movement'][1])))
        # Only apply limits once we have a full history
        foms[outside_limits & (n_missing_hsitory == 0)] += fom_threshold

    else:
        # No points yet assigned for this strike point so can't compare to history
        # Therefore just use amplitude to inform fom
        # Important this is sensible as following points will be weighted to follow this selection
        if n_peaks > 0:
            # TODO: Improve use of partial_hist_pen
            foms = (((scalings['amplitude']/np.clip(amplitude_peaks, 1e-3, None)) ** weightings['high_amplitude']) +
                    partial_hist_pen)
        else:
            foms = []

    foms = xr.DataArray(foms, coords=peaks_info['R_peaks'].coords, name='foms')

    return foms

def select_next_strike_point_peak(peaks_info, strike_points, strike_point_index, foms, t, fom_threshold=1e5):
    n_strike_points = len(strike_points.columns.levels[0])

    if t in strike_points.index:
        peak_inds_assigned = [strike_points.loc[t, (i, 'ind')] for i in np.arange(n_strike_points)]
        peak_inds_assigned = np.array(peak_inds_assigned)[~np.isnan(peak_inds_assigned)]
    else:
        peak_inds_assigned = []
    fom_order = np.argsort(foms)

    if (len(foms) == 0) or (np.all(foms > fom_threshold)) or (len(peak_inds_assigned) == len(foms)):
        ind = np.nan
        amplitude = np.nan
        s = np.nan
        r = np.nan
        fom = np.nan
    else:
        for peak_ind in np.array(fom_order):
            ind = peaks_info['ind_peaks'][peak_ind]
            if ind not in peak_inds_assigned:
                break  # Proceed with this ind_peak
            else:
                # TODO: If previously assigned peak is a MUCH better match, reassign
                continue  # Check next peak
        else:
            # All peaks assigned to previous strike points
            print(peak_inds_assigned, foms)
            raise NotImplementedError

        ind = peaks_info['ind_peaks'][peak_ind]
        amplitude = peaks_info['amplitude_peaks'][peak_ind]
        s = peaks_info['s_global_peaks'][peak_ind]  # TODO: Handle missing s_global?
        r = peaks_info['R_peaks'][peak_ind]
        fom = foms[peak_ind]

        # raise NotImplementedError

    # TODO: Remove ealier values that are inconsistent with current tren

    strike_points.loc[t, (strike_point_index, 'ind')] = ind
    strike_points.loc[t, (strike_point_index, 'amplitude')] = amplitude
    strike_points.loc[t, (strike_point_index, 's_global')] = s
    strike_points.loc[t, (strike_point_index, 'R')] = r
    strike_points.loc[t, (strike_point_index, 'fom')] = fom

    return strike_points


def sp_select_old():


    if n_peaks > strike_point_index:
        # Pick out nth local peak at this time slice
        ind_peaks_array = peak_order[strike_point_index]  # index of nth highest amplitude peak in peaks array
        ind_peak_i = ind_peaks[ind_peaks_array]  # Index of peak in full radial profile
        s_peak_i = s_peaks[ind_peaks_array]
        amplitude_peak_i = amplitude_peaks[ind_peaks_array]
    else:
        # No nth local peak at this time slice
        ind_peak_i = np.nan
        s_peak_i = np.nan
        amplitude_peak_i = np.nan
    if np.sum(~np.isnan(strike_points.loc[:, (strike_point_index, 'ind')])) == 0:
        # Assign first strike point location from which others should follow
        # Just use first nth highest local maximum
        strike_points.loc[coord, (strike_point_index, 'ind')] = ind_peak_i
        strike_points.loc[coord, (strike_point_index, 'amplitude')] = amplitude_peak_i
        strike_points.loc[coord, (strike_point_index, 's_global')] = s_peak_i
    else:
        if n_peaks > strike_point_index:
            if strike_point_assumptions == 'continuous':
                # Follow path of strike point by ensuring next point is close to previous location
                i_last_non_nan = int(np.max(np.nonzero(np.array(
                    ~strike_points.loc[:, (strike_point_index, 'ind')].isnull()))[0]))
                t_last_non_nan = strike_points.index[i_last_non_nan]

                ind_peak_prev = strike_points.loc[t_last_non_nan, (strike_point_index, 'ind')]
                s_peak_prev = strike_points.loc[t_last_non_nan, (strike_point_index, 's_global')]
                amplitude_peak_prev = strike_points.loc[t_last_non_nan, (strike_point_index, 'amplitude')]

                # Working with speak in space order - not sorted amplitude order
                amplitude_diffs = (amplitude_peaks - amplitude_peak_prev) / amplitude_peak_prev + 0.1
                s_diff_scale_factor = s_coord.max() - s_coord.min()  # TODO: move
                i_diffs = (ind_peak_i - ind_peak_prev)
                if np.any(np.abs(i_diffs) > 5):
                    pass
                s_diffs = (s_peaks - s_peak_prev) / s_diff_scale_factor + 0.01
                s_diffs[s_diffs < 0] += 0.1
                # TODO: Deal with perfect fom of 0 for # same s coord
                foms = (np.abs(amplitude_diffs) * np.abs(s_diffs)).values  # want low figure of merit
                i_foms = np.argsort(foms)

                for j in np.arange(len(i_foms)):
                    ind_peaks_array = int(i_foms[j])  # index of peak in peaks array with lowest fom
                    if ind_peaks_array not in peak_inds_assigned:
                        # Make sure best match for this strike point hasn't already been assigned to higher
                        # priority strike point
                        peak_inds_assigned.append(ind_peaks_array)
                        break
                else:
                    raise ValueError('Shouldnt get here..')
            else:
                raise NotImplementedError(f'strike_point_assumptions = {strike_point_assumptions}')

            # ind_peaks_array = ind_peaks[i_sp]
            ind_peak_i = ind_peaks_sorted[ind_peaks_array]
            s_peak_i = s_peaks[ind_peaks_array]
            r_peak_i = r_peaks[ind_peaks_array]
            amplitude_peak_i = amplitude_peaks[ind_peaks_array]
            fom = float(foms[ind_peaks_array])

            pass
        else:
            logger.debug(
                f'Time {coord} after initial strike point identification has {n_peaks}<{strike_point_index} local maxima')

            ind_peak_i = np.nan
            s_peak_i = np.nan
            r_peak_i = np.nan
            amplitude_peak_i = np.nan
            fom = np.nan

        strike_points.loc[coord, (strike_point_index, 'ind')] = ind_peak_i
        strike_points.loc[coord, (strike_point_index, 'amplitude')] = amplitude_peak_i
        strike_points.loc[coord, (strike_point_index, 's_global')] = s_peak_i
        strike_points.loc[coord, (strike_point_index, 'R')] = r_peak_i
        strike_points.loc[coord, (strike_point_index, 'fom')] = fom

    strike_points.loc[coord, ('summary', 'n_peaks')] = n_peaks
    coord_previous = coord

    peaks_info['strike_points'] = strike_points

    # peaks_info[f'peak'].append(amplitude_peak_global)
    # peaks_info[f'ind_global_peak'].append(peaks_info[f'ind_global_peak'])
    # peaks_info[f'n_peaks'].append(peaks_info[f'n_peaks'])
    # peaks_info[f'ind_peaks'].append(peaks_info[f'ind_peaks'])
    # peaks_info[f'peak_properties'][coord] = peaks_info[f'properties']
    # if coords is not None:
    #     for key, values in coords.items():
    #         peak = np.array(values)[i_peak_global] if n_peaks > 0 else np.nan
    #         peaks_info[f'{key}_peak'].append(peak)

def locate_strike_points(param_values, dim='t', coords=None, peak_kwargs=(('width', 3),),
                         strike_point_assumptions='continuous'):
    """Return value and location of global peak value, and sorted indices of other peaks.

    Args:
        param_values: 1D/2D array of values [t, R]
        dim:
        peak_kwargs: Dict of args to pass to scipy.signal.find_peaks

    Returns: Dict of peak info

    """
    logger.info('Calculating strike point locations')
    if isinstance(peak_kwargs, tuple):
        peak_kwargs = dict(peak_kwargs)
    if 'n' in param_values.dims:
        param_values = param_values.swap_dims({'n': 't'})
    param = param_values.name
    # ind_peaks = argrelmax(param_values.values, axis=1, order=order, mode='clip')
    peak_info = defaultdict(list)
    peak_info[f'peak_properties'] = {}
    # if coords is not None:
    #     for key in coords.keys():
    #         peak_info[f'{key}_peaks'] = []

    # n_strike_points = 4
    n_strike_points = 0



    # strike_point_ind = defaultdict(list)
    # strike_point_value = defaultdict(list)
    # cols = list(itertools.chain(*[[f'strike_point_{i}_ind', f'strike_point_{i}_peak']
    #                               for i in np.arange(n_strike_points)]))
    cols = [np.arange(n_strike_points), ['ind', 'R', 's_global', 'amplitude', 'fom']]
    cols = pd.MultiIndex.from_product(cols, names=['strike_point', 'param'])
    strike_points = pd.DataFrame(columns=cols, dtype=float)
    strike_points.index.name = 't'

    roll = param_values.rolling({dim: 1})  # Roll over single time slices
    for i_coord, (coord, profile) in enumerate(roll):
        profile = profile.sel({dim: coord}).values  # Make 1D
        coord = float(coord.values)

        peaks_info_i = find_peaks_info(profile, peak_kwargs=peak_kwargs, add_boundary_local_maxima=False, coords=coords)

        fom_threshold = 1e5
        for strike_point_index in np.arange(n_strike_points):
            # TODO: Pass in weightings from input file
            foms = calc_strike_point_continutity_rating(peaks_info_i, strike_points, strike_point_index, t=coord,
                                                        fom_threshold=fom_threshold)
            strike_points = select_next_strike_point_peak(peaks_info_i, strike_points, strike_point_index, foms,
                                                          t=coord, fom_threshold=fom_threshold)
            pass

        i_peak = peaks_info_i[f'ind_peak_global']
        n_peaks = peaks_info_i[f'n_peaks']

        peak_info[f'amplitude_peak_global'].append(peaks_info_i[f'amplitude_peak_global'])
        peak_info[f'ind_peak_global'].append(peaks_info_i[f'ind_peak_global'])
        peak_info[f'n_peaks'].append(peaks_info_i[f'n_peaks'])
        peak_info[f'ind_peaks'].append(peaks_info_i[f'ind_peaks'])
        peak_info[f'amplitude_peaks'].append(peaks_info_i[f'amplitude_peaks'])
        peak_info[f'peak_properties'][coord] = peaks_info_i[f'properties']
        peak_info[f'strike_points'] = strike_points
        if coords is not None:
            for key, values in coords.items():
                if n_peaks > 0:
                    peak = np.array(values)[i_peak]
                else:
                    peak = np.nan
                peak_info[f'{key}_peak'].append(peak)

    # func = np.vectorize(find_peaks)
    # ind_peaks = func(param_values.values)
    keys_1d = ([f'amplitude_peak_global', f'ind_peak_global', f'n_peaks'] +
               list(f'{key}_peak' for key in coords.keys()))
    for key in keys_1d:
        peak_info[key] = np.array(peak_info[key])

    return peak_info


def calc_2d_profile_param_stats(data, stats=('min', ('percentile', 1), 'mean', 'std', ('percentile', 99), 'max'),
                                coords_reduce='t', path=None, roll_width=None, roll_center=True,
                                roll_reduce_func='mean'):
    stats = make_iterable(stats)
    logger.info(f'Calculating temporal stats (roll={roll_width}): {stats}')
    out = xr.Dataset({})
    labels = {}

    # Move path string to end of var name ie append stat info to param name
    path_str = f'_{path}' if path else ''
    param_path = data.name
    param = param_path.replace(path_str, '')

    coord_keep, axis_keep, axis_reduce = get_reduce_coord_axis_keep(data, coords_reduce)
    coord_keep_str = coord_keep.replace(path_str, '')

    if roll_width is not None:
        data_in = data
        len_axis_keep = len(data[axis_keep])
        roll_width = roll_width if (len_axis_keep > roll_width) else len_axis_keep
        roll = data.rolling({coord_keep: roll_width}, center=True)
        coord_roll = coord_keep+'_win'
        data = roll.construct(coord_roll)
        # coords_reduce_rolled = (coord_keep, coord_roll)
        roll_str = f'_roll{roll_width:d}'
        # This opperation can be slow!
        data = getattr(data, roll_reduce_func)(dim=coord_roll)
    else:
        roll_str = ''
        data_rolled = data
        coords_reduce_rolled = coord_keep

    for stat in stats:
        if '%' in stat:
            n_percent = float(stat.replace('%', ''))
            stat = ('percentile', n_percent)

        if stat == 'min':
            key = f'{param}{roll_str}_min({coord_keep_str}){path_str}'
            out[key] = data.min(dim=coords_reduce)
            labels['min'] = key
        elif stat == 'mean':
            key = f'{param}{roll_str}_mean({coord_keep_str}){path_str}'
            out[key] = data.mean(dim=coords_reduce)
            labels['mean'] = key
        elif stat == 'std':
            key = f'{param}{roll_str}_std({coord_keep_str}){path_str}'
            out[key] = data.std(dim=coords_reduce)
            labels['std'] = key
        elif stat == 'max':
            key = f'{param}{roll_str}_max({coord_keep_str}){path_str}'
            out[key] = data.max(dim=coords_reduce)
            labels['max'] = key
        elif isinstance(stat, tuple) and stat[0] == 'percentile':
            key = f'{param}{roll_str}_{stat[1]}percentile({coord_keep_str}){path_str}'
            # out[f'{param}_{stat[1]}percentile({coord_keep_str}){path_str}'] = reduce_2d_data_array(data, np.nanpercentile,
            #                                                                                     coords_reduce, stat[1])
            out[key] = data.quantile(stat[1]/100, dim=coords_reduce).drop_vars('quantile')
            labels[f'{stat[1]:0.1f}%'] = key
        else:
            raise ValueError(stat)

    return out, labels

def calc_1d_profile_rolling_stats(data, stats=('mean', 'std'), path=None, width=11):

    out = xr.Dataset({})

    # Move path string to end of var name ie append stat info to param name
    path_str = f'_{path}' if path else ''
    param_path = data.name
    param = param_path.replace(path_str, '')
    coord = data.dims[0]

    width = width if (len(data) > width) else len(data)

    roll = data.rolling({coord: width})

    for stat in stats:
        if stat == 'min':
            out[f'{param}_roll-min({coord})_{path}'] = roll.min()
        elif stat == 'mean':
            out[f'{param}_roll-mean({coord})_{path}'] = roll.mean()
        elif stat == 'std':
            out[f'{param}_roll-std({coord})_{path}'] = roll.std()
        elif stat == 'max':
            out[f'{param}_roll-max({coord})_{path}'] = roll.max()
        elif isinstance(stat, tuple) and stat[0] == 'percentile':
            # out[f'{param}_{stat[1]}percentile({coord_keep_str})_{path}'] = xr.apply_ufunc(np.percentile, data, stat[1],
            #                                     input_core_dims=((coord_reduce,), ()), kwargs=dict(axis=axis_keep))
            out[f'{param}_roll-{stat[1]}percentile({coord})_{path}'] = np.percentile(roll, stat[1])  # Use apply_ufunc
        else:
            raise ValueError(stat)

    for var, values in out.items():
        values.attrs['rolling_stat_width'] = width

    return out

def calc_physics_params(path_data, path_name, params=None, meta_data=None):
    """Old IDL sched iranalysis.pro input parameters
      ;variables
      ;sh = shot
      ; trange= time range, array e.g [0.0,0.1]
      ;ldef - returned from get_ldef, which uses loc - the ldef path definitions file looked up with loc
      ;loc - analysis path name e.g. 'louvre4'
      ;t - returned times
      ;s - returned radius
      ;h - returned temperature?
      ;qpro - returned heat flux
      ;numsatpix - total number of saturated pixels on analysis path
      ;alphaconst - alpha constant, can be array, defined in rit2air_comb_view.pro
      ;tsmooth -
      ;tbgnd -
      ;targetrate -
      ;nline -
      ;aug - run for ASDEX, defunct?
      ;print - set flag for output to PS
    """
    if params is None:
        params = ['heat_flux', 'temperature', 'frame_data_nuc', 'frame_data']
        # params = ['heat_flux', 'temperature', 'frame_data_nuc']
    if meta_data is None:
        meta_data = {}

    path = path_name  # abbreviation for format strings

    data_out = path_data

    r_path = path_data[f'R_{path}']
    s_path = path_data[f's_global_{path}']
    annulus_areas_horizontal = calc_horizontal_path_anulus_areas(r_path)
    if False:
        # TODO: Collect tile angles from structure definition file
        # TODO: Move calc_tile_tilt_area_coorection_factors fucntion to mast_u machine plugins
        # tile_angle_poloidal = path_data[f'tile_angle_poloidal_{path}']
        # tile_angle_toroidal = path_data[f'tile_angle_toroidal_{path}']
        poloidal_plane_tilt = dict(T1=45, T2=45, T3=45, T4=0, T5=-45)
        # toroidal_tilt = dict(T1=4, T2=4, T3=4, T4=4, T5=4)
        step_size = dict(T1=0.003, T2=0.003, T3=0.003, T4=0.003, T5=0.0014)  # Tile edge step in metres
        nlouvres = dict(T1=12, T2=12, T3=12, T4=12, T5=24)
        try:
            tile_tilt_area_factors = calc_tile_tilt_area_correction_factors(path_data,
                                        poloidal_plane_tilt=poloidal_plane_tilt, nlouvres=nlouvres, step_size=step_size,
                                        path=path_name)
        except Exception as e:
            logger.warning(f'Failed to calculate calc_tile_tilt_area_correction_factors. Setting to 1.')
            tile_tilt_area_factors = np.ones_like(path_data[f'heat_flux_{path}'])

        # TODO: Fix erroneous negative and large correction factors eg +277.4440
        tile_tilt_area_factors[:, :] = 1


        annulus_areas_corrected = annulus_areas_horizontal * tile_tilt_area_factors
    else:
        logger.warning(f'Using incorrect annulus areas for integrated/total quantities')
        annulus_areas_corrected = np.tile(annulus_areas_horizontal, (len(data_out['t']), 1))
        # tile_tilt_area_factors = np.ones((len(data_out['t']), len(annulus_areas_horizontal)))
        tile_tilt_area_factors = np.ones_like(annulus_areas_corrected)

    data_out[f'tile_tilt_area_factors_{path}'] = (('t', f'i_{path}',), tile_tilt_area_factors)
    data_out[f'annulus_areas_{path}'] = (('t', f'i_{path}',), annulus_areas_corrected)
    annulus_areas_corrected = data_out[f'annulus_areas_{path}']

    for param in params:
        info_numeric = ['amplitude_peak_global', 'n_peaks', 's_global_peak', 'R_peak']
        try:
            param_values = path_data[f'{param}_{path}']
            param_total = calc_divertor_area_integrated_param(param_values, annulus_areas_corrected)
            prominence = (np.percentile(np.array(param_values), 99.9) - np.percentile(np.array(param_values), 2)) / 2e3
            # prominence = np.ptp(np.array(param_values)) / 2e2
            peak_kwargs = dict(width=1, height=np.mean(param_values.values), prominence=prominence)
            # print(f'Using peak detection window settings: {peak_kwargs}')
            param_stats, labels = calc_2d_profile_param_stats(param_values, path=path)

            if param == 'heat_flux':
                peak_info = locate_strike_points(param_values, dim='t', coords=dict(s_global=s_path, R=r_path),
                                                 peak_kwargs=peak_kwargs)
                # TODO: Use new strike point info in output
                # debug = True
                debug = False
                if debug and (param == 'heat_flux'):
                    ax, data, artist = debug_plots.debug_plot_profile_2d(path_data, param='heat_flux', t_range=None,
                                                                         show=False, robust_percentiles=(40, 99.8))
                    r = path_data['R_path0']
                    t = path_data['t']
                    i_global = peak_info['ind_global_peak']
                    nan_mask = ~np.isnan(i_global)
                    strike_points = peak_info['strike_points']

                    colors = ('k', 'b', 'orange', 'g')
                    lw = 2.5
                    for i, color in zip(np.arange(4), colors):
                        ax.plot(strike_points.loc[:, (i, 'R')], t, color=color, lw=lw, marker='o', markersize=2,
                                alpha=0.4)
                        lw *= 0.5

                    # for t, i_peaks_t in zip(path_data['t'], peak_info['ind_peaks']):
                    #     ax.plot(r[i_peaks_t], np.repeat(t.item(), len(i_peaks_t)), 'gx', ms=2, alpha=0.4)
                    # ax.plot(r[i_global[nan_mask].astype(int)], path_data['t'][nan_mask], 'kx', ms=2, alpha=0.4)
                    plt.show()

        except Exception as e:
            logger.warning(f'Failed to calculate physics summary for param "{param}":\n{e}')
            raise e
        else:
            data_out = data_out.merge(param_stats)

            key = f'{param}_total_{path}'
            data_out[key] = (('t',), np.array(param_total))
            data_out[key].attrs.update(meta_data.get(f'{param}_total', {}))
            if 'description' in data_out[key].attrs:
                data_out[key].attrs['label'] = data_out[key].attrs['description']

            for info in info_numeric:
                key = f'{param}_{info}_{path}'
                value = peak_info[info]  # Value for each time point, with nan for frames with no peaks
                data_out[key] = (('t',), value)
                data_out = attach_standard_meta_attrs(data_out, varname=key, key=f'{param}_{info}')
                # data_out[key].attrs.update(meta_data.get(f'{param}_{info}', {}))
                # if 'description' in data_out[key].attrs:
                #     data_out[key].attrs['label'] = data_out[key].attrs['description']
                if np.all(np.isnan(value)):
                    logger.warning(f'All values are nan for physics parameter: {key}')

        # Calculate rolling std and mean of peak positions
        try:
            key = f'{param}_R_peak_{path}'
            param_r_peak_values = data_out[key]
            param_stats = calc_1d_profile_rolling_stats(param_r_peak_values, stats=['mean', 'std'], path=path)
        except Exception as e:
            logger.warning(f'Failed to calculate physics summary for param "{key}":\n{e}')
            raise e
        else:
            data_out = data_out.merge(param_stats)
            # TODO: Attach attrs meta data?

    return data_out

def add_peak_shifted_coord(data_array, coord_new='{coord}_peak_shifted', coord_old=None):
    if coord_old is None:
        coord_old = data_array.dims[0]
    coord_new = coord_new.format(coord=coord_old, coord_old=coord_old)

    data_array.swap_dims({data_array.dims[0]: coord_old})

    coord_original = data_array[coord_old]

    offset_value = data_array.argmax(dim=coord_old)
    offset_value = coord_original[offset_value]

    coord_new_value = coord_original - offset_value

    data_array.coords[coord_new] = coord_new_value
    coord_new_value = data_array.coords[coord_new]
    coord_new_value.attrs.update(data_array.coords[coord_old].attrs)
    coord_new_value.attrs['long_name'] = coord_new_value.attrs.get('symbol', coord_old) + (f' (peak aligned)')

    data_array = data_array.swap_dims({data_array.dims[0]: coord_new})

    return data_array, coord_new

def add_custom_shifted_coord(data_array, coord_old, coord_new, data, offset_param='heat_flux_r_peak_path0', slice_=None):
    """Add new coordinate to supplied data_array by subtracting value from reference signal:
    coord_new = data_array[x_coord] - data[offset_param].sel(slice_)

    Args:
        data_array      : Xarray dataarray to add coordinate to
        coord_old       : Existing coordinate of data_array to apply shift to
        coord_new       : Name of new shifted coordinate to add to data_array
        data            : Dataset from which to extract offset param
        offset_param    : Parameter from data to apply as shift
        slice_:         : Optional slice to select offset param value

    Returns: data_array with new coord added

    """
    coord_original = data_array[coord_old]

    offset_value = data[offset_param]
    if slice_ is not None:
        offset_value = offset_value.sel(slice_)

    coord_new_value = coord_original - offset_value

    data_array.coords[coord_new] = coord_new_value

    data_array.swap_dims({data_array.dims[0]: coord_new})

    return data_array, coord_new

if __name__ == '__main__':
    pass

