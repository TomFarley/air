# -*- coding: future_fstrings -*-
#!/usr/bin/env python

"""


Created: 
"""

import logging
from typing import Union, Iterable, Tuple, Optional
from pathlib import Path

import numpy as np
import xarray as xr
import scipy.interpolate

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def dl_to_temerature(frame_data, calib_coefs, bb_curve, exposure, temp_nuc_bg=23):
    """Convert NUC corrected IR camera Digital Level (DL) values to temperatures in deg C.

    Args:
        frame_data  : 3D array of frame DL values (frame_no, ypix, xpix) TODO: Check pix order
        calib_coefs : Temperature calibration cooficients (a_grad, a_intcp, b_grad, b_intcp, window_trans)
        bb_curve    : 2 column array mapping temperatures in deg C to numbers of photons
        exposure    : Camera exposure time in us
        temp_nuc_bg : Temperature in deg C of uniform background subtracted in NUC correction (typically room temp)

    Returns: 3D array of frame data converted to temperatures in deg C

    """
    # temp_bg=23  #  background temperature
    bb_curve = bb_curve.reset_index()
    exposure = exposure * 1e-6

    cols_bb = set(bb_curve.columns)
    expected_cols_bb = set(('photon_flux', 'temperature_celcius'))
    if cols_bb != expected_cols_bb:
        raise ValueError(f'Black body curve data has unexpected format, columns: {cols_bb}. '
                         f'Expected columns: {expected_cols_bb}')

    # TODO: break into sub functions and improve variable naming
    c1 = (calib_coefs['a_grad'])/(exposure)+calib_coefs['a_intcp']
    c2 = (calib_coefs['b_grad'])/(exposure)+calib_coefs['b_intcp']
    trans_correction = calib_coefs['trans']  # window transmission correction (>1 as window attenuates)

    # convert counts to photons. Include the attenuation caused by the window
    frame_photons = c2*(frame_data*trans_correction)

    # use the data from bbtable.dat to convert from recorded photons to temp.
    # need to add on the photons which are the background (NUC makes zero ADC counts
    # the background temp
    f_photons = scipy.interpolate.interp1d(bb_curve['temperature_celcius'], bb_curve['photon_flux'], kind='linear')
    f_temp = scipy.interpolate.interp1d(bb_curve['photon_flux'], bb_curve['temperature_celcius'], kind='linear')
    phot_bg = f_photons(temp_nuc_bg)

    # then determine the temperature using the photon counts plus the bckg.
    frame_temps = xr.apply_ufunc(f_temp, frame_photons+phot_bg)
    frame_temps.name = 'frame_temps'
    frame_temps.attrs.update({'long_name': r'$T$',
                              'units': r'$^\circ C$',
                              'description': 'Surface temperature observed by each pixel'})
    return frame_temps

if __name__ == '__main__':
    pass