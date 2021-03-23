#!/usr/bin/env python

"""


Created: 
"""

import logging
from collections import defaultdict

import numpy as np
import xarray as xr
from fire.geometry.geometry import calc_horizontal_path_anulus_areas, calc_tile_tilt_area_coorection_factors, \
    calc_divertor_area_integrated_param
from fire.misc.data_structures import attach_standard_meta_attrs, get_reduce_coord_axis_keep, reduce_2d_data_array
from fire.misc.utils import make_iterable

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

def find_peaks_info(profile, peak_kwargs=()):
    from scipy.signal import find_peaks

    peak_info = defaultdict(list)

    if isinstance(peak_kwargs, tuple):
        # NOTE: width arg for find_peaks is measured smaller than expected - not same as order to argrelmax,
        # so best not to specify it...
        peak_kwargs = dict(peak_kwargs)  # (('width', 3),)
    ind_peaks, properties = find_peaks(profile, **peak_kwargs)
    # ind_peaks = argrelmax(profile, order=peak_kwargs['width'])

    # find peaks doesn't count nan values towards peak indices, so add them on? NO
    # nan_sum = np.cumsum(np.isnan(profile))
    # for i, i_peak in enumerate(copy(ind_peaks)):
    #     ind_peaks[i] += nan_sum[i_peak]

    # Order peaks in decending order
    peak_order = np.argsort(profile[ind_peaks])[::-1]
    ind_peaks = make_iterable(ind_peaks[np.array(peak_order)], cast_to=np.ndarray)
    n_peaks = len(ind_peaks)

    # Global maxima
    if n_peaks > 0:
        i_peak_global = ind_peaks[0]
        param_peak = profile[i_peak_global]
    else:
        i_peak_global = np.nan
        param_peak = np.nan

    peak_info[f'peak_value'] = param_peak
    peak_info[f'ind_global_peak'] = i_peak_global
    peak_info[f'n_peaks'] = n_peaks
    peak_info[f'ind_peaks'] = ind_peaks
    peak_info[f'peak_properties'] = properties

    if isinstance(profile, xr.DataArray):
        coord = profile[profile.dims[0]]
        peak_info['x_peaks'] = coord[ind_peaks]
        peak_info['x_global_peak'] = coord[i_peak_global]

    return peak_info

def locate_target_peak(param_values, dim='t', coords=None, peak_kwargs=(('width', 3),)):
    """Return value and location of global peak value, and sorted indices of other peaks.

    Args:
        param_values: 1D/2D array of values [t, R]
        dim:
        peak_kwargs: Dict of args to pass to scipy.signal.find_peaks

    Returns: Dict of peak info

    """
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

    roll = param_values.rolling({dim: 1})  # Roll over single time slices
    for coord, profile in roll:
        profile = profile.sel({dim: coord}).values  # Make 1D
        coord = float(coord.values)

        peak_info_i = find_peaks_info(profile, peak_kwargs)
        # ind_peaks, properties = find_peaks(profile, **peak_kwargs)
        #
        # # Order peaks in decending order
        # peak_order = np.argsort(profile[ind_peaks])[::-1]
        # ind_peaks = ind_peaks[peak_order]
        # n_peaks = len(ind_peaks)
        #
        # # Global maxima
        # if n_peaks > 0:
        #     i_peak = ind_peaks[0]
        #     param_peak = profile[i_peak]
        # else:
        #     i_peak = np.nan
        #     param_peak = np.nan

        i_peak = peak_info_i[f'ind_global_peak']
        n_peaks = peak_info_i[f'n_peaks']

        peak_info[f'peak'].append(peak_info_i[f'peak_value'])
        peak_info[f'ind_global_peak'].append(peak_info_i[f'ind_global_peak'])
        peak_info[f'n_peaks'].append(peak_info_i[f'n_peaks'])
        peak_info[f'ind_peaks'].append(peak_info_i[f'ind_peaks'])
        peak_info[f'peak_properties'][coord] = peak_info_i[f'properties']
        if coords is not None:
            for key, values in coords.items():
                peak = np.array(values)[i_peak] if n_peaks > 0 else np.nan
                peak_info[f'{key}_peak'].append(peak)

    # func = np.vectorize(find_peaks)
    # ind_peaks = func(param_values.values)
    keys_1d = ([f'peak', f'ind_global_peak', f'n_peaks'] +
               list(f'{key}_peak' for key in coords.keys()))
    for key in keys_1d:
        peak_info[key] = np.array(peak_info[key])

    return peak_info


def calc_2d_profile_param_stats(data, stats=('min', ('percentile', 1), 'mean', 'std', ('percentile', 99), 'max'),
                                coord_reduce='t', path=None):

    out = xr.Dataset({})

    # Move path string to end of var name ie append stat info to param name
    path_str = f'_{path}' if path else ''
    param_path = data.name
    param = param_path.replace(path_str, '')

    # # Get preserved dimension used for xr.apply_ufunc
    # if data.values.ndim == 2:
    #     coord_keep = list(data.dims)
    #     coord_keep.pop(data.dims.index(coord_reduce))
    #     coord_keep = coord_keep[0]
    #     axis_keep = list(data.dims).index(coord_keep)
    #     coord_keep_str = coord_keep.replace(path_str, '')
    # else:
    #     coord_keep = None
    #     axis_keep = None
    #     coord_keep_str = ''
    coord_keep, axis_keep, axis_reduce = get_reduce_coord_axis_keep(data, coord_reduce)
    coord_keep_str = coord_keep.replace(path_str, '')

    for stat in stats:
        if stat == 'min':
            out[f'{param}_min({coord_keep_str})_{path}'] = data.min(dim=coord_reduce)
        elif stat == 'mean':
            out[f'{param}_mean({coord_keep_str})_{path}'] = data.mean(dim=coord_reduce)
        elif stat == 'std':
            out[f'{param}_std({coord_keep_str})_{path}'] = data.std(dim=coord_reduce)
        elif stat == 'max':
            out[f'{param}_max({coord_keep_str})_{path}'] = data.max(dim=coord_reduce)
        elif stat == 'mean':
            out[f'{param}_mean({coord_keep_str})_{path}'] = data.mean(dim=coord_reduce)
        elif isinstance(stat, tuple) and stat[0] == 'percentile':
            # out[f'{param}_{stat[1]}percentile({coord_keep_str})_{path}'] = xr.apply_ufunc(np.percentile, data, stat[1],
            #                                     input_core_dims=((coord_reduce,), ()), kwargs=dict(axis=axis_keep))
            out[f'{param}_{stat[1]}percentile({coord_keep_str})_{path}'] = reduce_2d_data_array(data, np.percentile,
                                                                                                coord_reduce, stat[1])
        else:
            raise ValueError(stat)

    return out

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
        # params = ['heat_flux', 'temperature', 'frame_data_nuc', 'frame_data']
        params = ['heat_flux', 'temperature', 'frame_data_nuc']
    if meta_data is None:
        meta_data = {}

    path = path_name  # abbreviation for format strings

    data_out = path_data

    r_path = path_data[f'R_{path}']
    s_path = path_data[f's_global_{path}']
    annulus_areas_horizontal = calc_horizontal_path_anulus_areas(r_path)
    if True:
        # TODO: Collect tile angles from structure definition file
        # TODO: Move calc_tile_tilt_area_coorection_factors fucntion to mast_u machine plugins
        # tile_angle_poloidal = path_data[f'tile_angle_poloidal_{path}']
        # tile_angle_toroidal = path_data[f'tile_angle_toroidal_{path}']
        poloidal_plane_tilt = dict(T1=45, T2=45, T3=45, T4=0, T5=-45)
        toroidal_tilt = dict(T1=4, T2=4, T3=4, T4=4, T5=4)
        nlouvres = dict(T1=12, T2=12, T3=12, T4=12, T5=24)  # Not required
        tile_tilt_area_factors = calc_tile_tilt_area_coorection_factors(path_data,
                                    poloidal_plane_tilt=poloidal_plane_tilt, toroidal_tilt=toroidal_tilt,
                                                                        nlouvres=nlouvres, path=path_name)
        annulus_areas_corrected = annulus_areas_horizontal * tile_tilt_area_factors
    else:
        logger.warning(f'Using incorrect annulus areas for integrated/total quantities')
        annulus_areas_corrected = annulus_areas_horizontal

    data_out[f'annulus_areas_{path}'] = ((f'i_{path}',), annulus_areas_corrected)
    annulus_areas_corrected = data_out[f'annulus_areas_{path}']

    for param in params:
        info_numeric = ['peak', 'n_peaks', 's_peak', 'r_peak']
        try:
            param_values = path_data[f'{param}_{path}']
            param_total = calc_divertor_area_integrated_param(param_values, annulus_areas_corrected)
            peak_info = locate_target_peak(param_values, dim='t', coords=dict(s=s_path, r=r_path))
            param_stats = calc_2d_profile_param_stats(param_values, path=path)
        except Exception as e:
            logger.warning(f'Failed to calculate physics summary for param "{param}":\n{e}')
            raise e
        else:
            data_out = data_out.merge(param_stats)

            key = f'{param}_total_{path}'
            data_out[key] = (('t',), param_total)
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
            key = f'{param}_r_peak_{path}'
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

