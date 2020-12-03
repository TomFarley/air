#!/usr/bin/env python

"""


Created: 
"""

import logging, time, re
from typing import Union, Iterable, Tuple, Optional
from pathlib import Path

import numpy as np
import xarray as xr

from fire.misc.data_quality import calc_outlier_nsigma_for_sample_size

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
        nsigma = calc_outlier_nsigma_for_sample_size(data.size)
    thresh = np.nanmean(data) + nsigma * np.nanstd(data)
    if np.nanmax(data) < thresh:
        out = np.max(data)
    else:
        # Look for peaks in tail of histogram
        n = 3
        # Generate list of percentiles weighted towards high values
        x = np.power(np.linspace(20, 100**n, len(data)/sample_size_factor), 1/n)
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
    coord_path = f'i_{path}'
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
                new_key = re.sub('_im$', f'_{path}', key)
            else:
                new_key = f'{key}_{path}'
            # TODO: Handle 't' as active dim name
            coords = ('n',)
            coords = tuple((coord for coord in coords if coord in data.dims)) + (coord_path,)
            data_path_extracted[new_key] = (coords, data.sel(x_pix=x_pix_path, y_pix=y_pix_path))
            data_path_extracted[new_key].attrs.update(data.attrs)  # Unnecessary?

    # Set alternative coordinates to index path data (other than path index)
    alternative_path_coords = ('R', 's', 's_global', 'phi')  # , 'x', 'y', 'z')
    for coord in alternative_path_coords:
        coord = f'{coord}_{path}'
        if coord in data_path_extracted:
            data_path_extracted = data_path_extracted.assign_coords(
                                                    **{coord: (coord_path, data_path_extracted[coord].values)})

    if ('n' in data_path_extracted.dims):
        data_path_extracted['n'] = image_data['n']
    return data_path_extracted


if __name__ == '__main__':
    pass