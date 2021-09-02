#!/usr/bin/env python

"""


Created: 
"""

import logging
from typing import Union, Iterable, Sequence, Tuple, Optional, Any, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

def calc_field_of_view(focal_length, pixel_pitch=30e-6, image_shape=(320, 256)):
    """Calculate angular field of view [degrees] and solid angle of view [steradians] for a pixel and whole image.

    Solid angles are calculated for a four-sided right rectangular pyramid.

    Args:
        focal_length: Focal length of lens in meters
        pixel_pitch: Pixel (square) dimension in meters
        image_shape: Pixel resolution of image

    Returns: (Dict of values for a pixel, Dict of (tuple) values for image)

    """
    image_shape = np.array(image_shape)

    if focal_length == "":
        logger.warning('Empty string for lens focal length. Assuming default value of 25 mm')
        focal_length = 25e-3

    fov_pixel = np.arctan2(pixel_pitch, focal_length)  # Angular field of view of (square) pixel in radians
    solid_angle_pixel = 4*np.arcsin(np.sin(fov_pixel/2)**2)  # Solid angle viewed by a single pixel
    frac_2pi_pixel = solid_angle_pixel/(2*np.pi)

    fov_image = fov_pixel * image_shape  # Field of view for each dimension of image
    solid_angle_image = 4*np.arcsin(np.sin(fov_image[0]/2)*np.sin(fov_image[1]/2))  # Solid angle of whole image view
    frac_2pi_image = solid_angle_image/(2*np.pi)

    detector_dimensions= image_shape * pixel_pitch
    detector_area = np.product(detector_dimensions)
    fov_pixel = np.rad2deg(fov_pixel)
    fov_image = np.rad2deg(fov_image)

    fov_info = dict(fov_pixel=fov_pixel, solid_angle_pixel=solid_angle_pixel, frac_2pi_pixel=frac_2pi_pixel,
                    fov_image=fov_image, solid_angle_image=solid_angle_image, frac_2pi_image=frac_2pi_image,
                    detector_dimensions=detector_dimensions, detector_area=detector_area)
    return fov_info

def calc_photon_flux_correction_for_focal_length_change(focal_length_calib, focal_length_actual,
                                                        pixel_pitch=30e-6, image_shape=(320, 256)):
    if isinstance(focal_length_calib, str):
        parts = focal_length_calib.split(' ')
        if parts[1] == 'mm':
            focal_length_calib = float(parts[0])*1e-3
        else:
            raise ValueError()

    fov_info_calib = calc_field_of_view(focal_length_calib, pixel_pitch=pixel_pitch, image_shape=image_shape)
    fov_info_actual = calc_field_of_view(focal_length_actual, pixel_pitch=pixel_pitch, image_shape=image_shape)

    flux_correction_factor = fov_info_actual['solid_angle_pixel'] / fov_info_calib['solid_angle_pixel']
    flux_correction_factor = (focal_length_actual/focal_length_calib)**-2
    if flux_correction_factor != 1:
        logger.warning(f'Focal length correction factor is not unity: {flux_correction_factor}, '
                       f'lens_calib={focal_length_calib}m, lens_actual={focal_length_actual}m')

    return flux_correction_factor

if __name__ == '__main__':
    pass