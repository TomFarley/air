#!/usr/bin/env python

"""


Created: 
"""

import logging, time, re
from typing import Union, Iterable, Tuple, Optional
from pathlib import Path
from datetime import datetime
import collections

import numpy as np
import xarray as xr
from scipy import interpolate

import matplotlib.pyplot as plt

from fire.misc import data_quality
from fire.misc import data_structures

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

def median_filter_anisotropic(images, blur_axis=1):
    """Get IR detector bands"""
    # Blur along bands in image to blur bad pixels but not bands
    # footprint_size = 5
    # footprint = np.zeros((footprint_size, footprint_size))
    # footprint[int(footprint_size/2), :] = 1

    n_repeat = images.shape[blur_axis]
    blurred_image = np.tile(np.median(images, axis=blur_axis), (n_repeat, 1)).T
    return blurred_image

def median_filter(images, blur_axis=None, kwargs=(('size', 3), ('mode', 'mirror'))):
    t0 = datetime.now()
    if blur_axis is None:
        from scipy.ndimage import median_filter  # from scipy.signal import medfilt
        from skimage.filters.rank import median
        kwargs = dict(kwargs)

        kws = dict(size=3, mode='mirror')
        kws.update(kwargs)

        size = kwargs.get('size', 3)
        if (images.ndim == 3) and (not isinstance(size, (list, tuple, np.ndarray))):
            kws['size'] = (1, size, size)  # Don't filter across frames
        if (images.ndim == 2) and (np.min(images) >= 0):
            # More efficient, but only for 2D image
            footprint = np.ones((kws['size'], kws['size']))
            images_blured = median(np.array(images).astype(np.uint16), selem=footprint)
        else:
            images_blured = median_filter(images, **kws)  # images_blured = medfilt(images, **kwargs)
    else:
        images_blured = median_filter_anisotropic(images, blur_axis=blur_axis)

    t1 = datetime.now()
    # logger.info(f'Applied median filter in {(t1-t0).total_seconds():0.1f}s')
    return images_blured

def identify_bad_pixels(frame_data, method='blur_diff', n_sigma_tol=3, n_bad_pixels_expected=None, blur_axis=1):

    if method == 'blur_diff':
        if n_bad_pixels_expected is not None:
            n_pixels_image = np.prod(frame_data.shape[-2:])
            n_sigma_tol = data_quality.calc_outlier_nsigma_for_sample_size(n_pixels_image,
                                                                           n_outliers_expected=n_bad_pixels_expected)

        # Remove IR detector bands
        detector_bands = median_filter(frame_data, blur_axis=blur_axis)
        detector_bands -= np.min(detector_bands)  # Just want the band variation - not the absolute intensity of frame
        images_without_bands = frame_data - detector_bands

        images_blured = median_filter(images_without_bands, blur_axis=None, kwargs=dict(size=7, mode='mirror'))
        # TODO: When passed multiple frames average along frame axis to get hot pixel image
        images_hot_pixels = images_without_bands - images_blured

        threshold_hot = 0 + n_sigma_tol * np.std(images_hot_pixels)

        # Find the hot pixels, but ignore the edges
        mask_bad_pixels = (np.abs(images_hot_pixels) > threshold_hot)
    elif method == 'flicker_offset':
        mask_bad_pixels = find_dead_pixels(frame_data=frame_data, threshold_for_bad_difference=13,
                                           threshold_for_bad_low_std=0, threshold_for_bad_std=10)
    else:
        raise ValueError(f'{method} not recognised')
    bad_pixels = np.nonzero(mask_bad_pixels)
    # bad_pixels = np.nonzero((np.abs(images_without_bands_hot_pixels[1:-1, 1:-1]) > threshold))
    bad_pixels = np.array(bad_pixels)  # + 1  # because we ignored the first row and first column

    n_bad_pixels = bad_pixels.shape[-1]
    logger.info(f'Identified {n_bad_pixels} bad pixels')

    return bad_pixels, mask_bad_pixels, threshold_hot, detector_bands, images_blured

def bpr_list_to_mask(bpr_list, detector_window):
    """Convert list of bad pixel coordinates to an image mask which is True where the pixels are bad"""
    bpr_list = np.array(bpr_list)

    if bpr_list.shape[-1] == 2:
        bpr_list = bpr_list.T

    # Calcam conventions:
    #  Order: (Left,Top,Width,Height)
    #  Index: Left/Top etc start from 0 (as opposed to 1 for IPX index conventions)
    left, top, width, height = detector_window

    mask_bpr = np.zeros((height, width), dtype=bool)
    mask_bpr[bpr_list[0]-top, bpr_list[1]-left] = True
    # mask_bpr = mask_bpr[::-1]  # Not sure if needed?

    logger.debug(f'BPR mask needs checking - doesnt seem to align with bad pixels or BPR already applied?')

    return mask_bpr

def apply_bpr_correction(frame_data, mask_bad_pixels, blurred_data=None, method='median_loop',
                         kwargs=(('size', 3), ('mode', 'mirror'))):
    """Return images with the hot pixels removed"""
    bad_pixels = np.array(np.nonzero(mask_bad_pixels))
    # bad_pixels = xr.DataArray(bad_pixels, dims=['y_pix', 'x_pix'])
    bad_pixels_y = xr.DataArray(np.array(bad_pixels[0]), dims=['x_pix'])
    bad_pixels_x = xr.DataArray(np.array(bad_pixels[1]), dims=['x_pix'])

    n_bad_pixels = bad_pixels.shape[-1]
    if n_bad_pixels == 0:  # No pixels to fix so don't perform slow median filter etc
        logger.debug(f'No bad pixels to replace')
        return frame_data
    t0 = datetime.now()

    if method == 'median':
        if blurred_data is None:
            blurred_data = median_filter(frame_data, blur_axis=None, kwargs=kwargs)
            blurred_data = data_structures.movie_data_to_dataarray(blurred_data,
                                frame_times=frame_data.t, frame_nos=frame_data.n, name='frame_data_blured')
        fixed_images = frame_data  # np.copy(frame_data)

        #TODO: Speed up by looping over bad pixles and applying median filter only to bad pixel areas
        fixed_images[..., bad_pixels_y, bad_pixels_x] = blurred_data[..., bad_pixels_y, bad_pixels_x]
    elif method == 'median_loop':
        fixed_images = replace_dead_pixels(frame_data, mask_bad_pixels=mask_bad_pixels)
    else:
        raise NotImplementedError

    t1 = datetime.now()
    logger.info(f'Applied Bad Pixel Replacement (BPR) for {n_bad_pixels} pixels in {(t1-t0).total_seconds():0.1f}s '
                f'using "{method}" filter ')

    return fixed_images

def find_dead_pixels(frame_data, start_interval='auto', end_interval='auto', framerate='auto', threshold_for_bad_low_std=0,
                     threshold_for_bad_std=10, threshold_for_bad_difference=13):
    """Based on Fabio Federici function:
    https://github.com/ciozzaman/IRVB/blob/3e2c2b4ff31388e4f7dd5c9eb455f581b891337a/python_library/collect_and_eval/
    collect_and_eval/__init__.py#L4144"""

    # Created 26/02/2019
    # function that finds the dead pixels. 'data' can be a string with the path of the record or the data itself.
    # if the path is given the oscillation filtering is done
    # default tresholds for std and difference found 25/02/2019

    # Output legend:
        # 3 = treshold for difference with neighbouring pixels trepassed
        # 6 = treshold for std trepassed
        # 9 = both treshold trepassed

    frame_data = np.array(frame_data)

    if end_interval == 'auto':
        end_interval = np.shape(frame_data)[1]
    else:
        if framerate == 'auto':
            print('if you specify an end time you must specify the framerate')
            exit()
        else:
            end_interval = end_interval * framerate
            if (end_interval > np.shape(frame_data)[1]):
                end_interval = np.shape(frame_data)[1]

    if start_interval=='auto':
        start_interval=0
    else:
        if framerate=='auto':
            print('if you specify a start time you must specify the framerate')
            exit()
        else:
            start_interval = start_interval * framerate
            if (start_interval > end_interval):
                print("You set the start interval after the end one, I'm going to use zero seconds")
                start_interval = 0


    frame_data = frame_data[0, start_interval:end_interval]

    # in /home/ffederic/work/irvb/vacuum_chamber_testing/Aug13_2018/irvb_sample-000018
    # I found that some frames can be completely missing, filled with 0 or 64896
    # and this messes with calculating std and mean, so I need to remove this frames
    min_data = np.nanmin(frame_data, axis=(-1, -2))
    # max_data = np.nanmin(data,axis=-1,-2)		# I don't want to use this because saturation could cause this too
    frame_data = frame_data[min_data > 0]

    mean = np.mean(frame_data, axis=0)
    std = np.std(frame_data, axis=0)
    flag_check = np.zeros(np.shape(std))
    flag = np.ones(np.shape(std))
    # for i in range(np.shape(std)[0]):
    # 	for j in range(np.shape(std)[1]):
    # 		if std[i,j]>treshold_for_bad_std:
    # 			flag_check[i,j]=6
    # 			flag[i, j] = 0
    flag_check[std > threshold_for_bad_std] = 6
    flag[std > threshold_for_bad_std] = 0
    flag_check[std <= threshold_for_bad_low_std] = 6
    flag[std <= threshold_for_bad_low_std] = 0
    for repeats in [0, 1]:  # Two passes to catch adjacent bad pixels
        for i in range(1, np.shape(std)[0] - 1):
            for j in range(1, np.shape(std)[1] - 1):
                if flag_check[i,j] in [3,9]:
                    continue
                temp = (mean[i - 1, j - 1:j + 2] * flag[i - 1, j - 1:j + 2]).tolist() + [
                    (mean[i, j - 1] * flag[i, j - 1]).tolist()] + [(mean[i, j + 1] * flag[i, j + 1]).tolist()] + (
                               mean[i + 1, j - 1:j + 2] * flag[i + 1, j - 1:j + 2]).tolist()
                if len(temp) == 0:
                    # flag_check[i, j] += 3
                    # flag[i, j] = 0
                    continue
                else:
                    temp2 = [x for x in temp if x != 0]
                    if len(temp2) == 0:
                        # flag_check[i, j] += 3
                        # flag[i, j] = 0
                        continue
                    else:
                        if (mean[i, j] > max(temp2) + threshold_for_bad_difference or mean[i, j] < min(
                                temp2) - threshold_for_bad_difference):
                            flag_check[i, j] += 3
                            flag[i, j] = 0
    # follows a slower version that checks also the edges
    # i_indexes = (np.ones_like(std).T*np.linspace(0,np.shape(std)[0]-1,np.shape(std)[0])).T
    # j_indexes = np.ones_like(std) * np.linspace(0, np.shape(std)[1] - 1, np.shape(std)[1])
    # # gna=np.zeros_like(std)
    # for repeats in [0, 1]:
    # 	for i in range(np.shape(std)[0] ):
    # 		for j in range(np.shape(std)[1] ):
    # 			if flag_check[i,j] in [3,9]:
    # 				continue
    # 			temp = mean[np.logical_and(flag , np.logical_and(np.abs(i_indexes-i)<=1 , np.logical_and( np.abs(j_indexes-j)<=1 , np.logical_or(i_indexes!=i , j_indexes!=j))))]
    # 			# gna[i,j] = np.std(temp)
    # 			if (mean[i, j] > max(temp) + treshold_for_bad_difference or mean[i, j] < min(temp) - treshold_for_bad_difference):
    # 				flag_check[i, j] += 3
    # 				flag[i, j] = 0


    counter = collections.Counter(flatten_full(flag_check))
    print('Number of pixels that trepass ' + str(threshold_for_bad_difference) + ' counts difference with neighbours: ' + str(counter.get(3)))
    print('Number of pixels with standard deviation > ' + str(threshold_for_bad_std) + ' counts: ' + str(counter.get(6)))
    print('Number of pixels that trepass both limits: '+str(counter.get(9)))

    return flag_check


def replace_dead_pixels(frame_data, mask_bad_pixels):

    # Created 26/02/2019, Fabio Federici
    # function replace dead pixels with the median/average from their neighbours

    frame_data = np.array(frame_data)

    mask_bad_pixels = np.array(mask_bad_pixels)

    if frame_data.shape[-2:] != mask_bad_pixels.shape:
        raise ValueError('The shape of the frame data and dead pixels mask do not match')

    frame_data_corrected = np.array(frame_data)

    for i_y in np.arange(np.shape(frame_data)[-2]):
        for i_x in np.arange(np.shape(frame_data)[-1]):
            if mask_bad_pixels[i_y, i_x] != 0:
                neighbour_values = []
                for i_y_offset in [-1, 0, 1]:
                    for i_x_offset in [-1, 0, 1]:
                        if (i_y_offset != 0 and i_x_offset != 0):
                            y_pix = np.clip(i_y + i_y_offset, 0, mask_bad_pixels.shape[0]-1)
                            x_pix = np.clip(i_x + i_x_offset, 0, mask_bad_pixels.shape[1]-1)

                            if mask_bad_pixels[y_pix, x_pix] == 0:  # make sure neighbour is good
                                # print(i_y, i_x, y_pix, x_pix)
                                neighbour_values.append(frame_data[..., y_pix, x_pix])
                frame_data_corrected[..., i_y, i_x] = np.median(neighbour_values, axis=0)

    return frame_data_corrected

def find_outlier_pixels(images, n_sigma_tol=3, check_edges=True, blur_axis=1):
    """Find the hot or dead pixels in a 2D dataset.
    NOTE: Symetric median blur method does not deal well with band structures from IR images - need

    Taken from https://stackoverflow.com/questions/18951500/automatically-remove-hot-dead-pixels-from-an-image-in-python

    Args:
        images:
        n_sigma_tol           : number of standard deviations used to cutoff the hot pixels
        check_edges   : ignore the edges and greatly speed up the code

    Returns: (list of hot pixels, image with with hot pixels removed)

    """

    t0 = time.time()

    bad_pixels, mask_bad_pixels, threshold_hot, detector_bands, images_blured = identify_bad_pixels(images, n_sigma_tol=n_sigma_tol, blur_axis=blur_axis)

    if check_edges == True:
        height, width = np.shape(images)

        ###Now get the pixels on the edges (but not the corners)###

        #left and right sides
        for index in range(1,height-1):
            #left side:
            med  = np.median(images[index - 1:index + 2, 0:2])
            diff = np.abs(images[index, 0] - med)
            if diff>threshold_hot:
                hot_pixels = np.hstack(( bad_pixels, [[index],[0]]  ))
                images_blured[index,0] = med

            #right side:
            med  = np.median(images[index - 1:index + 2, -2:])
            diff = np.abs(images[index, -1] - med)
            if diff>threshold_hot:
                hot_pixels = np.hstack(( bad_pixels, [[index],[width-1]]  ))
                images_blured[index,-1] = med

        #Then the top and bottom
        for index in range(1,width-1):
            #bottom:
            med  = np.median(images[0:2, index - 1:index + 2])
            diff = np.abs(images[0, index] - med)
            if diff>threshold_hot:
                hot_pixels = np.hstack(( bad_pixels, [[0],[index]]  ))
                images_blured[0,index] = med

            #top:
            med  = np.median(images[-2:, index - 1:index + 2])
            diff = np.abs(images[-1, index] - med)
            if diff>threshold_hot:
                hot_pixels = np.hstack(( bad_pixels, [[height-1],[index]]  ))
                images_blured[-1,index] = med

        ###Then the corners###

        #bottom left
        med  = np.median(images[0:2, 0:2])
        diff = np.abs(images[0, 0] - med)
        if diff>threshold_hot:
            hot_pixels = np.hstack(( bad_pixels, [[0],[0]]  ))
            images_blured[0,0] = med

        #bottom right
        med  = np.median(images[0:2, -2:])
        diff = np.abs(images[0, -1] - med)
        if diff>threshold_hot:
            hot_pixels = np.hstack(( bad_pixels, [[0],[width-1]]  ))
            images_blured[0,-1] = med

        #top left
        med  = np.median(images[-2:, 0:2])
        diff = np.abs(images[-1, 0] - med)
        if diff>threshold_hot:
            hot_pixels = np.hstack(( bad_pixels, [[height-1],[0]]  ))
            images_blured[-1,0] = med

        #top right
        med  = np.median(images[-2:, -2:])
        diff = np.abs(images[-1, -1] - med)
        if diff>threshold_hot:
            hot_pixels = np.hstack(( bad_pixels, [[height-1],[width-1]]  ))
            images_blured[-1,-1] = med
    t1 = time.time()
    logger.debug(f'Found {len(hot_pixels)} outlier pixels for frame in {t1-t0:0.3f}s')
    return hot_pixels, images_blured

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
    alternative_path_coords = ('R', 's', 's_global', 'phi', 'z')  # , 'x', 'y', 'z')
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

    # TODO: Move renaming of coords (eg 'i_in_frame_path0' -> 'i_path0') to dedicated function
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