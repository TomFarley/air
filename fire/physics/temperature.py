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
from scipy.constants import zero_Celsius

from fire.physics.black_body import calc_photons_to_temperature
from fire.camera.field_of_view import calc_photon_flux_correction_for_focal_length_change

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

def dl_to_photon_count_legacy(frame_data, calib_coefs, exposure, trans_correction=None):
    """In the old MAST scheduler, rather than calculating the photon count arriving at the detector,
    the output from the camera was rescaled to match a fixed lookup table. Therefore:
    - Rather than multiplying the photon counts by the exposure time, the camera counts were effectively divided by
        the exposure time to get the DL if exposure had been for a whole second.
    - Rather than multiplying the photon counts by the transmision factor, the camera counts were scaled up by
        dividing by the "trans_correction" = 1/(transmission coeff)
    """
    if 'b_grad' in calib_coefs:
        # Use original IDL temperature calibration file format
        # Following convention of IDL code for naming c1, c2  variables (inconsistent for air and ait)
        c1 = (calib_coefs['a_grad'])/(exposure)+calib_coefs['a_intcp']  # Not used
        c2 = (calib_coefs['b_grad'])/(exposure)+calib_coefs['b_intcp']
        trans_correction = calib_coefs['trans']  # window transmission correction (>1 as window attenuates)
    else:
        # Use FIRE temperature calibration format
        c2 = (calib_coefs['c1_grad'])/(exposure)+calib_coefs['c1_intcp']
        trans_correction = 1.61  # MAST air value - should be 1.3 for ait

    # Convert counts to photons. Include the attenuation caused by the window
    frame_photons = c2*(frame_data*trans_correction)

    return frame_photons

def dl_to_temerature_legacy(frame_data, calib_coefs, bb_curve, exposure, solid_angle_pixel, trans_correction,
                            temperature_bg_nuc=23, meta_data=None):
    if meta_data is None:
        meta_data = {}

    bb_curve = bb_curve.reset_index()

    if exposure in ('Unknown', None):
        exposure = 20e-6
        logger.warning(f'In absence of exposure value, using arbitrary value for debugging purposes: {exposure}us')

    cols_bb = set(bb_curve.columns)
    expected_cols_bb = set(('photon_flux', 'temperature_celcius'))
    if cols_bb != expected_cols_bb:
        raise ValueError(f'Black body curve data has unexpected format, columns: {cols_bb}. '
                         f'Expected columns: {expected_cols_bb}')

    # TODO: break into sub functions and improve variable naming

    frame_photons = dl_to_photon_count_legacy(frame_data, calib_coefs, exposure, trans_correction)

    # use the data from bbtable.dat to convert from recorded photons to temp.
    # need to add on the photons which are the background (NUC makes zero ADC counts the background temp
    f_photons = scipy.interpolate.interp1d(bb_curve['temperature_celcius'], bb_curve['photon_flux'], kind='linear')
    f_temp = scipy.interpolate.interp1d(bb_curve['photon_flux'], bb_curve['temperature_celcius'], kind='linear')
    phot_bg = f_photons(temperature_bg_nuc)

    # then determine the temperature using the photon counts plus the bckg.
    frame_temperature = xr.apply_ufunc(f_temp, frame_photons+phot_bg)

    frame_temperature.name = 'frame_temperature'
    frame_temperature.attrs.update(meta_data.get('temperature', {}))
    if 'description' in frame_temperature.attrs:
        frame_temperature.attrs['label'] = frame_temperature.attrs['description']
    return frame_temperature

def dl_to_excess_photon_flux(frame_data, c1, c2=0.0, c0=0.0):
    """Convert camera DL ADC counts to number of photons arriving at detector (2pi sr rather than pixel FOV) per second
    that are above the counts due to the background (NUC) temperature. ie excess to background/NUC temperature.
    NOTE: This is not the number photons arriving at the detector in an integration time as the c coefficients
    should already have been scaled by integration time.

    Args:
        frame_data: NUC subtracted camera data

        c1: Linear coefficient - photons/count/s [photons/s/m^2]
        c2: Quadratic coefficient - photons/count^2/s [photons/s/m^2]
        c0: Zero order coefficient (Not currently used)
    Returns: 'Excess photons' x 1/integration time for each pixel

    """


    # Convert DL counts to photons arriving at the detector, using linear sensitivity of sensor.
    # Calculation of calibration coefficients for MAST-U cameras is described in IR_calibration_final_20190219.docx
    # and performed by the script perform_calib.pro
    # Coefficient C_1 is found from a linear fit to photons_excess vs DL_counts, where
    # photons_excess=(ϕ_DL-ϕ̅_bckg) * t_int  is the number of photons additional to background, scaled by exposure time
    # ie plotting (Photons-Photons_BG)*t_int vs DL_counts, where Photons_BG is
    # calculated as the mean of the intercepts of plots of photons vs counts for each integration time
    frame_photon_flux_excess = c2*frame_data**2 + c1*frame_data + c0

    return frame_photon_flux_excess

def get_photon_coefs_from_calib_data(calib_coefs, integration_time):
    """Convert camera calibration data to photon flux conversion coefficients.
    ϕ_excess = c2 I_(DL,excess)^2 + c1 I_(DL,excess) + c0   (5)
    c coefficients are stored as photons*t_int/DL ie accumulated photons per exposure time per Digital Level.
    Therefore, to get excess photon flux (photons/s) we must divide by exposure time.
    The full excess photon flux can then be added to be background photon flux to look up the surface temperature.

    Args:
        calib_coefs: Calibration coefficients as read from FIRE calibration file
        integration_time: camera integration/exposure_time [s]

    Returns: Dict of photon conversion coefficients

    """
    if (calib_coefs['c1'] is not None):
        c1 = calib_coefs['c1'] / (integration_time*1e6)  # /1e6  # Enhancements cameras calibrated in us
    elif (calib_coefs['c1_grad'] is not None) and (calib_coefs['c1_intcp'] is not None):
        c1 = (calib_coefs['c1_grad']) / (integration_time) + calib_coefs['c1_intcp']
    else:
        raise ValueError(f'Calibration data does not contain values for c1 coefficient: {calib_coefs}')

    if (calib_coefs['c2'] is None):
        c2 = 0
        logger.info(f'Set quadratic temperature calibration coefficient to zero: c2=0. Only have value for c1.')
    elif (calib_coefs['c2_intcp'] is not None):
        c2 = (calib_coefs['c2_grad']) / (integration_time) + calib_coefs['c2_intcp']
    else:
        c2 = calib_coefs['c2'] / integration_time

    if ('c0' in calib_coefs) and (calib_coefs['c0'] is not None):
        c0 = calib_coefs['c0'] / integration_time
    else:
        c0 = 0

    return dict(c2=c2, c1=c1, c0=c0)

def get_background_photons(photon_lookup_table, temperature_bg_nuc_celcius=23, calib_coefs=None, use_calib_bg_value=False):
    if use_calib_bg_value:
        # Use background photon count from lab calibration
        photons_background = calib_coefs['photons_bg']
    else:
        # Interpolate lookup table to get photon count at NUC background temperature
        temperatures, photons = photon_lookup_table['temperature_celcius'], photon_lookup_table['n_photons']
        f_photons = scipy.interpolate.interp1d(temperatures, photons, kind='linear')
        photons_background = f_photons(temperature_bg_nuc_celcius)
    if isinstance(photons_background, np.ndarray) and (photons_background.size == 1):
        photons_background = photons_background.item()
    return photons_background

def photons_to_temperature_celcius(frame_photons, photon_lookup_table, to_celcius=True):
    photons = photon_lookup_table['n_photons']
    temperatures = photon_lookup_table['temperature_celcius'] if to_celcius else photon_lookup_table['temperature']
    f_temp = scipy.interpolate.interp1d(photons, temperatures, kind='linear')
    frame_temperature = xr.apply_ufunc(f_temp, frame_photons)
    return frame_temperature

def dl_to_temerature(frame_data_nuc, calib_coefs, wavelength_range, integration_time, transmittance=1.0,
                     solid_angle_pixel=2*np.pi, lens_focal_length=25e-3, temperature_bg_nuc=23,
                     temperature_range=(0,800), meta_data=None, use_calib_bg_value=False):
    """Convert NUC corrected IR camera Digital Level (DL) values to temperatures in deg C.

    Args:
        frame_data_nuc  : 3D array of frame DL values after nuc subtraction (frame_no, ypix, xpix) TODO: Check pix order
        calib_coefs : Dict of temperature calibration coefficients (a_grad, a_intcp, b_grad, b_intcp, window_trans)
        wavelength_range    : Wavelength range to integrate photon flux over
        integration_time    : Camera exposure time in us
        transmittance       : Transmittance of optics (eg windows, lens) between surface and detector in range [0,1]
        solid_angle_pixel   : Solid angle viewed by a single pixel (calculated in calc_field_of_view)
        lens_focal_length   : Focal length of lens used for correction from lab calibration lens focal length
        temperature_bg_nuc : Temperature in deg C of uniform background subtracted in NUC correction (typically room temp)
        temperature_range: Temperature range to calculate photon conversion table over
        meta_data:

    Returns: 3D array of frame data converted to temperatures in deg C

    """
    if meta_data is None:
        meta_data = {}

    # ******************************************************************
    # temp_bg=23  #  background temperature
    # bb_curve = bb_curve.reset_index()

    # TODO: Remove tmp range
    # temperature_range_kelvin = bb_curve['temperature_celcius'] + zero_Celsius  # TMP
    # ******************************************************************

    # if True:
    #     import matplotlib.pyplot as plt
    #     for n in np.arange(len(frame_data)):
    #         plt.figure(num=n)
    #         plt.imshow(frame_data[n])
    #         plt.colorbar()
    #         plt.show()

    temperature_range_kelvin = np.arange(*temperature_range, 1) + zero_Celsius

    # Calculate lookup table of photon fluxes (ie integration_time=1.0s) for range of temperatures
    photons_lookup_table, _, _ = calc_photons_to_temperature(temperature_range_kelvin, wavelength_range=wavelength_range,
                                                    emissivity=1, solid_angle=solid_angle_pixel,
                                                    integration_time=1, wavelength_step=1.5e-8, transmittance=1)
    # Convert calibration data to photon conversion coefficients
    photon_coefs = get_photon_coefs_from_calib_data(calib_coefs, integration_time=integration_time)
    # Calculate excess photon flux above room temperature
    frame_photon_flux_excess = dl_to_excess_photon_flux(frame_data_nuc, **photon_coefs)

    # TODO: Pass in correct window transmittance
    # Divide by transmittance as calculating photons emitted by surface rather than recieved at detector
    frame_photon_flux_excess = frame_photon_flux_excess / transmittance

    # from fire.plotting.image_figures import annimate_image_data
    # annimate_image_data(frame_data)

    # Calculate room temperature photon flux
    # use_calib_bg_value = True
    photon_flux_nuc_bg = get_background_photons(photons_lookup_table, temperature_bg_nuc_celcius=temperature_bg_nuc,
                                            calib_coefs=calib_coefs, use_calib_bg_value=use_calib_bg_value)
    # Total photons arriving at detector
    frame_photon_flux_total = frame_photon_flux_excess + photon_flux_nuc_bg

    # Account for difference in lens compared to that used for lab calibration - CHANGE OF LENS HAS NO EFFECT
    # focal_length_correction_factor = calc_photon_flux_correction_for_focal_length_change(calib_coefs['lens'],
    #                                                                             focal_length_actual=lens_focal_length)
    # frame_photon_flux_total *= focal_length_correction_factor

    # Look up temperatures for each pixel's photon count
    frame_temperature = photons_to_temperature_celcius(frame_photon_flux_total, photons_lookup_table)

    if False:
        # TODO: Remove/move debug plotting to debug_plots module
        logger.info(f'Plotting temperature images for {len(frame_temperature)} images')
        import matplotlib.pyplot as plt
        for n in np.arange(len(frame_temperature)):
            plt.figure(num=n)
            plt.imshow(frame_temperature[n])
            plt.colorbar()
            plt.show()

    # Add meta data  # TODO: remove as now done outside function
    # frame_temperature.name = 'frame_temperature'
    # frame_temperature.attrs.update(meta_data.get('temperature', {}))
    # if 'description' in frame_temperature.attrs:
    #     frame_temperature.attrs['label'] = frame_temperature.attrs['description']

    return frame_temperature

if __name__ == '__main__':
    pass