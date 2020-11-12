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


def locate_target_peak(param_values, dim='t', coords=None, peak_kwargs=None):
    """Return value and location of global peak value, and sorted indices of other peaks.

    Args:
        param_values: 1D/2D array of values [t, R]
        dim:
        peak_kwargs: Dict of args to pass to scipy.signal.find_peaks

    Returns: Dict of peak info

    """
    from scipy.signal import argrelmax, find_peaks
    if peak_kwargs is None:
        peak_kwargs = dict(width=3)
    if 'n' in param_values.dims:
        param_values = param_values.swap_dims({'n': 't'})
    param = param_values.name
    # ind_peaks = argrelmax(param_values.values, axis=1, order=order, mode='clip')
    peak_info = defaultdict(list)
    peak_info[f'peak_properties'] = {}
    # if coords is not None:
    #     for key in coords.keys():
    #         peak_info[f'{key}_peaks'] = []

    roll = param_values.rolling({dim: 1})
    for coord, profile in roll:
        profile = profile.sel({dim: coord}).values  # Make 1D
        coord = float(coord.values)
        ind_peaks, properties = find_peaks(profile, **peak_kwargs)

        # Order peaks in decending order
        peak_order = np.argsort(profile[ind_peaks])[::-1]
        ind_peaks = ind_peaks[peak_order]
        n_peaks = len(ind_peaks)

        # Global maxima
        if n_peaks > 0:
            i_peak = ind_peaks[0]
            param_peak = profile[i_peak]

        else:
            i_peak = np.nan
            param_peak = np.nan


        peak_info[f'peak'].append(param_peak)
        peak_info[f'ind_global_peak'].append(i_peak)
        peak_info[f'n_peaks'].append(n_peaks)
        peak_info[f'ind_peaks'].append(ind_peaks)
        peak_info[f'peak_properties'][coord] = properties
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
        params = ['heat_flux', 'temperature']
    if meta_data is None:
        meta_data = {}

    path = path_name  # abbreviation for format strings

    data_out = path_data

    r_path = path_data[f'R_{path}']
    s_path = path_data[f's_global_{path}']
    annulus_areas_horizontal = calc_horizontal_path_anulus_areas(r_path)
    if False:
        tile_angle_poloidal = path_data[f'tile_angle_poloidal_{path}']
        tile_angle_toroidal = path_data[f'tile_angle_toroidal_{path}']
        tile_tilt_area_factors = calc_tile_tilt_area_coorection_factors()
        annulus_areas_corrected = annulus_areas_horizontal * tile_tilt_area_factors
    else:
        logger.warning(f'Using incorrect annulus areas for integrated/total quantities')
        annulus_areas_corrected = annulus_areas_horizontal

    data_out[f'annulus_areas_{path}'] = ((f'i_{path}',), annulus_areas_corrected)
    annulus_areas_corrected = data_out[f'annulus_areas_{path}']

    for param in params:
        try:
            param_vales = path_data[f'{param}_{path}']
            param_total = calc_divertor_area_integrated_param(param_vales, annulus_areas_corrected)
            peak_info = locate_target_peak(param_vales, dim='t', coords=dict(s=s_path, r=r_path))
        except Exception as e:
            logger.warning(f'Failed to calculate physics summary for param "{param}":\n{e}')
            raise e
        else:
            key = f'{param}_total_{path}'
            data_out[key] = (('t',), param_total)
            data_out[key].attrs.update(meta_data.get(f'{param}_total', {}))
            if 'description' in data_out[key].attrs:
                data_out[key].attrs['label'] = data_out[key].attrs['description']

            info_numeric = ['peak', 'n_peaks', 's_peak', 'r_peak']
            for info in info_numeric:
                key = f'{param}_{info}_{path}'
                value = peak_info[info]
                data_out[key] = (('t',), value)
                data_out[key].attrs.update(meta_data.get(f'{param}_{info}', {}))
                if 'description' in data_out[key].attrs:
                    data_out[key].attrs['label'] = data_out[key].attrs['description']
                if np.all(np.isnan(value)):
                    logger.warning(f'All values are nan for physics parameter: {key}')

    return data_out

if __name__ == '__main__':
    pass

