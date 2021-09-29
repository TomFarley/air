#!/usr/bin/env python

"""


Created: 
"""

import logging, time, re
from typing import Union, Iterable, Tuple, Optional
from pathlib import Path

import numpy as np
import xarray as xr
from scipy import interpolate

from fire.misc import data_quality
from fire.misc import data_structures

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


def find_outlier_pixels(image, tol=3, check_edges=True):
    """Find the hot or dead pixels in a 2D dataset.

    Taken from https://stackoverflow.com/questions/18951500/automatically-remove-hot-dead-pixels-from-an-image-in-python

    Args:
        image:
        tol           : number of standard deviations used to cutoff the hot pixels
        check_edges   : ignore the edges and greatly speed up the code

    Returns: (list of hot pixels, image with with hot pixels removed)

    """

    from scipy.ndimage import median_filter
    t0 = time.time()
    blurred = median_filter(image, size=2)
    difference = image - blurred
    threshold = tol * np.std(difference)

    # Find the hot pixels, but ignore the edges
    hot_pixels = np.nonzero((np.abs(difference[1:-1, 1:-1]) > threshold))
    hot_pixels = np.array(hot_pixels) + 1  # because we ignored the first row and first column

    # Image with the hot pixels removed
    fixed_image = np.copy(image)
    for y, x in zip(hot_pixels[0], hot_pixels[1]):
        fixed_image[y, x] = blurred[y, x]

    if check_edges == True:
        height, width = np.shape(image)

        ###Now get the pixels on the edges (but not the corners)###

        #left and right sides
        for index in range(1,height-1):
            #left side:
            med  = np.median(image[index - 1:index + 2, 0:2])
            diff = np.abs(image[index, 0] - med)
            if diff>threshold:
                hot_pixels = np.hstack(( hot_pixels, [[index],[0]]  ))
                fixed_image[index,0] = med

            #right side:
            med  = np.median(image[index - 1:index + 2, -2:])
            diff = np.abs(image[index, -1] - med)
            if diff>threshold:
                hot_pixels = np.hstack(( hot_pixels, [[index],[width-1]]  ))
                fixed_image[index,-1] = med

        #Then the top and bottom
        for index in range(1,width-1):
            #bottom:
            med  = np.median(image[0:2, index - 1:index + 2])
            diff = np.abs(image[0, index] - med)
            if diff>threshold:
                hot_pixels = np.hstack(( hot_pixels, [[0],[index]]  ))
                fixed_image[0,index] = med

            #top:
            med  = np.median(image[-2:, index - 1:index + 2])
            diff = np.abs(image[-1, index] - med)
            if diff>threshold:
                hot_pixels = np.hstack(( hot_pixels, [[height-1],[index]]  ))
                fixed_image[-1,index] = med

        ###Then the corners###

        #bottom left
        med  = np.median(image[0:2, 0:2])
        diff = np.abs(image[0, 0] - med)
        if diff>threshold:
            hot_pixels = np.hstack(( hot_pixels, [[0],[0]]  ))
            fixed_image[0,0] = med

        #bottom right
        med  = np.median(image[0:2, -2:])
        diff = np.abs(image[0, -1] - med)
        if diff>threshold:
            hot_pixels = np.hstack(( hot_pixels, [[0],[width-1]]  ))
            fixed_image[0,-1] = med

        #top left
        med  = np.median(image[-2:, 0:2])
        diff = np.abs(image[-1, 0] - med)
        if diff>threshold:
            hot_pixels = np.hstack(( hot_pixels, [[height-1],[0]]  ))
            fixed_image[-1,0] = med

        #top right
        med  = np.median(image[-2:, -2:])
        diff = np.abs(image[-1, -1] - med)
        if diff>threshold:
            hot_pixels = np.hstack(( hot_pixels, [[height-1],[width-1]]  ))
            fixed_image[-1,-1] = med
    t1 = time.time()
    logger.debug(f'Found {len(hot_pixels)} outlier pixels for frame in {t1-t0:0.3f}s')
    return hot_pixels, fixed_image

def find_outlier_intensity_threshold(data, nsigma='auto', sample_size_factor=5):
    data = np.array(data).ravel()
    if nsigma == 'auto':
        nsigma = data_quality.calc_outlier_nsigma_for_sample_size(data.size)
    thresh = np.nanmean(data) + nsigma * np.nanstd(data)
    if np.nanmax(data) < thresh:
        out = np.max(data)
    else:
        # Look for peaks in tail of histogram
        n = 3
        # Generate list of percentiles weighted towards high values
        x = np.power(np.linspace(20, 100**n, int(len(data)/sample_size_factor)), 1/n)
        y = np.nanpercentile(data, x)
        d = np.concatenate([[0], np.diff(y)])
        # d2 = np.concatenate([[0], np.diff(d)])
        # Identify big jumps between percentiles
        jumps = np.where(d > np.max(d)/2)[0]
        out = y[jumps[-1]-1]
    return out


def extract_path_data_from_images(image_data: xr.Dataset, path_data: xr.Dataset,
                                  x_path='x_pix_{path}', y_path='y_pix_{path}', path_name='path0',
                                  keys=None):
    path = path_name
    x_path = x_path.format(path=path_name)
    y_path = y_path.format(path=path_name)

    in_frame_str = '_in_frame'
    coord_path = f'i{in_frame_str}_{path}'
    if coord_path not in path_data.coords:
        in_frame_str = ''
        coord_path = f'i{in_frame_str}_{path}'

    if keys is None:
        keys = image_data.keys()
    frame_shape = image_data['frame_data'].shape[1:]
    data_path_extracted = xr.Dataset(coords=path_data.coords)
    x_pix_path = path_data[x_path]
    y_pix_path = path_data[y_path]
    for key in keys:
        data = image_data[key]
        if (data.shape == frame_shape) or (data.shape[1:] == frame_shape):
            if re.match('.*_im$', key):
                new_key = re.sub('_im$', f'{in_frame_str}_{path}', key)
                key_var = re.sub('_im$', '', key)
            else:
                new_key = f'{key}{in_frame_str}_{path}'
                key_var = key
            # TODO: Handle 't' as active dim name
            coords = ('n', 'i_digitiser',)  # Possible additional coordinates
            coords = tuple((coord for coord in coords if coord in data.dims)) + (coord_path,)
            try:
                data_path_extracted[new_key] = (coords, np.array(data.sel(x_pix=x_pix_path, y_pix=y_pix_path)))
            except ValueError as e:
                raise e
            data_path_extracted[new_key].attrs.update(data.attrs)  # Unnecessary?
            data_path_extracted = data_structures.attach_standard_meta_attrs(data_path_extracted, varname=new_key,
                                                                             replace=True, key=key_var)
            if new_key == f's_global{in_frame_str}_{path}':
                # nans in s_global coord cause problems so interpolate/extrapolate them out
                s_values = data.sel(x_pix=x_pix_path, y_pix=y_pix_path)
                r_values = image_data[f'R_im'].sel(x_pix=x_pix_path, y_pix=y_pix_path)
                # Spline fit requires strictly increasing x
                i_r_order = np.argsort(r_values).values
                # Filter out constant values
                mask_unique_r = np.concatenate([[True], np.diff(r_values[i_r_order]) > 0])
                i_r_order_unique = i_r_order[mask_unique_r]
                mask_s_ordered_unique_nan = np.isnan(s_values[i_r_order_unique]).values
                i_ordered_unique_nan = i_r_order_unique[mask_s_ordered_unique_nan]
                i_ordered_unique_nonan = i_r_order_unique[~mask_s_ordered_unique_nan]
                if np.any(mask_s_ordered_unique_nan):
                    logger.debug(
                        'ds values contain nans - interpolating and extrapolating to fill nan values based on dR')
                    if np.sum(~mask_s_ordered_unique_nan) >= 2:
                        f = interpolate.InterpolatedUnivariateSpline(r_values[i_ordered_unique_nonan],
                                                                     s_values[i_ordered_unique_nonan],
                                                                     k=1, ext=0)
                        s_values[i_ordered_unique_nan] = f(r_values[i_ordered_unique_nan])
                        # Where there are repeated s values fill in with the preceding value to catch remaining nans
                        for i_duplicate in i_r_order[~mask_unique_r]:
                            # Need to loop to prevent nans being passed along adjacent duplicates
                            s_values[i_duplicate] = s_values[i_duplicate-1]
                        new_key_with_nans = f'{key_var}{in_frame_str}_with_nans_{path}'
                        data_path_extracted[new_key_with_nans] = data_path_extracted[new_key]
                        data_path_extracted[new_key][:] = s_values

    # Set alternative coordinates to index path data (other than path index)
    alternative_path_coords = ('R', 's', 's_global', 'phi')  # , 'x', 'y', 'z')
    for coord in alternative_path_coords:
        coord = f'{coord}{in_frame_str}_{path}'
        if coord in data_path_extracted:
            attrs = data_path_extracted[coord].attrs  # attrs get lost in conversion to coordinate
            data_path_extracted = data_path_extracted.assign_coords(
                                                    **{coord: (coord_path, data_path_extracted[coord].values)})
            data_path_extracted[coord].attrs.update(attrs)

    if ('n' in data_path_extracted.dims):
        data_path_extracted['n'] = image_data['n']
    return data_path_extracted

def filter_unknown_materials_from_analysis_path(path_data, path_name, missing_material_key=-1):
    path = path_name

    in_frame_str = '_in_frame'
    material_id_key = f'material_id{in_frame_str}_{path}'
    if material_id_key not in path_data.data_vars:
        in_frame_str = ''
        material_id_key = f'material_id{in_frame_str}_{path}'
    coord_i_path = f'i{in_frame_str}_{path}'

    material_id = path_data[material_id_key]
    if missing_material_key is not None:
        mask_unknown_material = (material_id == missing_material_key)
    else:
        mask_unknown_material = np.zeros_like(material_id, dtype=bool)

    mask_known_material = ~mask_unknown_material
    coords_i_path_known_mat = f'i_{path}'


    path_data[f'mask_known_material{in_frame_str}_{path}'] = (coord_i_path, np.array(mask_known_material))

    for coord in path_data.coords:
        if coord_i_path not in path_data[coord].dims:
            continue

        coord_name_known_mat = coord.replace(in_frame_str, '')
        coord_key = 'i_path'
        data_known_mat = path_data[coord].sel({coord_i_path: mask_known_material})
        attrs = path_data[coord].attrs
        path_data[coord_name_known_mat] = (coords_i_path_known_mat, np.array(data_known_mat))
        path_data = path_data.assign_coords(**{coord_name_known_mat:
                                                   (coords_i_path_known_mat, path_data[coord_name_known_mat].values)})
        path_data[coord_name_known_mat].attrs.update(attrs)
        path_data = data_structures.attach_standard_meta_attrs(path_data, varname=coord_name_known_mat,
                                                               replace=False, key=coord_key)

    # Make sure new i path index starts at 1 and is unity spaced
    coords_known_mat = {coords_i_path_known_mat: (coords_i_path_known_mat, np.arange(np.sum(mask_known_material)))}
    path_data = path_data.assign_coords(**coords_known_mat)

    for var_name in path_data.data_vars:
        if coord_i_path not in path_data[var_name].coords:
            continue
        # Variables without details in name meet all filter criteria ie in_frame, known_material etc
        var_name_known_mat = var_name.replace(in_frame_str, '')
        data_known_mat = path_data[var_name].sel({coord_i_path: mask_known_material})
        attrs = path_data[var_name].attrs
        coords = [coord if coord != coord_i_path else coords_i_path_known_mat
                    for coord in path_data[var_name].dims]
        path_data[var_name_known_mat] = (coords, np.array(data_known_mat))
        path_data[var_name_known_mat].attrs.update(attrs)
        path_data = data_structures.attach_standard_meta_attrs(path_data, varname=var_name_known_mat, replace=False)

        # TODO store data with separate coordinate for unknown material sections?

    return path_data

if __name__ == '__main__':
    pass