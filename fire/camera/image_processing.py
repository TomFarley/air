#!/usr/bin/env python

"""


Created: 
"""

import logging, time
from typing import Union, Iterable, Tuple, Optional
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
    logger.debug(f'Found outlier pixels for frame in {t1-t0:0.3f}s')
    return hot_pixels, fixed_image

if __name__ == '__main__':
    pass