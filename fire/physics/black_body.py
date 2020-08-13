#!/usr/bin/env python

"""


Created: 
"""

import logging

import numpy as np
from scipy import constants
from scipy.integrate import simps, romb
from scipy.constants import zero_Celsius
import pandas as pd

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

wlr_flir = np.array([4.1e-6, 5.0e-6])
wlr_ircam = np.array([7.7e-6, 9.4e-6])
wlr_sbf = np.array([4.5e-6, 5.0e-6])  #
wlr_thmsk = np.array([7.6e-6, 8.9e-6])  #

def planck_curve_wavelength_spectral_radiance(wavelengths, temperatures):
    """Return black body spectral radiance at given wavelengths and temperatures.

    Spectral radiance: 𝐼(𝜆,𝑇)=  (2ℎ𝑐^2)/𝜆^5   1/(𝑒^(ℎ𝑐/𝜆𝑘𝑇  )−1)    [W sr^-1 m^-2 m^-1]
    
    Args:
        wavelengths: Array of wavelengths (length M) [m]
        temperatures: Array of temerpatures (length N) [K]

    Returns: MxN array of number of BB photons emitted at each wavelength and temperature [photons s^-1 m^-2]

    """
    h = constants.Planck  # 6.63e-34
    c = constants.c  # 3.0e8
    k = constants.Boltzmann  # 1.38e-23

    lamda = wavelengths
    T = temperatures

    # TODO: Modify dimensions of arrays - np.meshgrid
    if np.array(lamda).shape != np.array(T).shape:
        T, lamda = np.meshgrid(T, lamda)

    assert np.array(lamda).shape == np.array(T).shape

    radiance = ((2 * h * c**2) / lamda**5) * (1 / (np.exp((h * c) / (lamda * k * T))-1))

    return radiance

def planck_curve_wavelength_photons(wavelengths, temperatures):
    """Flux of photons over a hemisphere emitted in a given wavelength range.

    Start with Spectral radiance, 𝐼(𝜆,𝑇)=  (2ℎ𝑐^2)/𝜆^5   1/(𝑒^(ℎ𝑐/𝜆𝑘𝑇  )−1)    [W sr-1m-3]
    Divide by the energy of one photon (hc/𝜆) and integrate over 2pi solid angle assuming Lambertian surfance (giving
    single factor of pi).
    Then for given wavelength range:
    𝜙_𝑝ℎ𝑜𝑡𝑜𝑛𝑠= 𝜖∫_(𝜆_1)^(𝜆_2) 2𝜋𝑐/𝜆^4 1/(𝑒^(ℎ𝑐/𝜆𝑘𝑇  )−1)

    Args:
        wavelengths: Array of wavelengths (length M) [m]
        temperatures: Array of temerpatures (length N) [K]

    Returns: MxN array of number of BB photons emitted at each wavelength and temperature [photons s^-1 m^-2]

    """
    h = constants.Planck  # 6.63e-34
    c = constants.c  # 3.0e8
    k = constants.Boltzmann  # 1.38e-23
    pi = constants.pi  # 3.14

    lamda = wavelengths
    T = temperatures

    # TODO: Modify dimensions of arrays - np.meshgrid
    if np.array(lamda).shape != np.array(T).shape:
        T, lamda = np.meshgrid(T, lamda)

    assert np.array(lamda).shape == np.array(T).shape

    n_photons = ((2 * pi * c) / lamda**4) * (1 / (np.exp((h * c) / (lamda * k * T))-1))

    return n_photons

def photons_psicic(wavelength, temp_kelvin):
    # differs by factor of pi
    expo_term = np.exp((h * c) / (wavelength * kb * temp_kelvin[i]))
    photons = emissivity_2D[i, :] * optical_transmittance * ((2 * c) / (wavelength ** 4)) * (1 / (expo_term - 1))
    return photons

def calc_wiens_displacment_law_wavelength(temperature):
    """Return wavelength [m] of peak in BB curve for given temperature [K] BB"""
    b = 2.897771955e-3   # m⋅K
    lamda = b/ temperature
    return lamda

def calc_photons_to_temperature(temperatures, wavelength_range, emissivity, integration_time, solid_angle=2*np.pi,
                                wavelength_step=1e-8, transmittance=1, temperature_error=None):
    """
    Calculate lookup table for number of photons emitted by surface at a given temp by integrating the blackbody curve

    Output depends on the integration time, emissivity, transmittance

    MWIR: lambda_range = [4.5e-6, 5.0e-6];
    LWIR: lambda_range = [7.6e-6, 8.9e-6];
     - taken from Elise's 2010 PSI paper.

    Args:
        temperatures: Temperatures in Kelvin [K]
        wavelength_range: Wavelength range in meters [m]
        emissivity: Shape of emissivity 0D, 1D array vs T (or lambda) or 2D array vs T and lambda.
        integration_time: Integration time in seconds [s]
        solid_angle: Solid angle in steradians [sr]
        temperature_error: If supplied, will return additional lookup tables at T+/- temperature_error

    Returns: (photons_total, n_photons_high, n_photons_low) - Each a pd.DataFrame lookup table.
             n_photons_high and n_photons_low low will be None if temperature_error=None

    """
    assert np.all(temperatures > 0), 'Supplied temperatures should be in Kelvin. Negative temperatures found.'
    if np.any(temperatures < 150):
        logger.warning(f'Received temperatures below 150K. Check T not passed in Celcius: {temperatures}')
    # TODO: Account for (wavelength dependent) transmittance here?

    wavelengths = np.arange(wavelength_range[0], wavelength_range[1], wavelength_step)  # (findgen(1001) / 1000) * 8e-6

    n_photons = planck_curve_wavelength_photons(wavelengths, temperatures)

    # Integrate photons over wavelength
    try:
        # Use higher accuracy Romberg Integration if have 2^k + 1 equally spaced wavelength samples
        photons_total = romb(n_photons, dx=wavelength_step, axis=0)
    except ValueError as e:
        photons_total = simps(n_photons, dx=wavelength_step, axis=0)

    # Apply scale factors
    # TODO: Handle wavelength/temperature dependent ie vector scale factors eg emissivity  (see PSICIC)
    # TODO: Use lens to calculate solid angle from rectangular field of view? Ω = || sin(φ) dθ dφ for cone
    scale_factor = emissivity * transmittance * integration_time * (solid_angle/(2*np.pi))
    photons_total = scale_factor * photons_total

    photons_total = pd.DataFrame(dict(temperature=temperatures, temperature_celcius=temperatures-zero_Celsius,
                                      n_photons=photons_total))
    photons_total.set_index('temperature')

    if temperature_error is not None:
        temperatures_high = temperatures + temperature_error
        temperatures_low = temperatures - temperature_error
        # TODO: Pass other args
        n_photons_high, _, _ = calc_photons_to_temperature(wavelengths, temperatures_high)
        n_photons_low, _, _ = calc_photons_to_temperature(wavelengths, temperatures_low)
    else:
        n_photons_high, n_photons_low = None, None

    return photons_total, n_photons_high, n_photons_low

def lookup_temperature_for_photon_count(temperature_lookup_table, photon_flux):
    from scipy import interpolate
    assert (np.all(photon_flux < np.max(temperature_lookup_table['n_photons'])) and
            np.all(photon_flux > np.min(temperature_lookup_table['n_photons']))), 'Photon flux outside lookup range'
    f = interpolate.interp1d(temperature_lookup_table['n_photons'], temperature_lookup_table['temperature'])
    temperatures = f(photon_flux)
    return temperatures

def calc_bg_temperatures_for_calibration():
    from fire.interfaces.interfaces import read_csv

    temperatures = np.arange(-100, 1450, 1) + zero_Celsius
    wlr_flir = np.array([4.1e-6, 5.0e-6])
    wlr_ircam = np.array([7.7e-6, 9.4e-6])
    wlr_sbf = np.array([4.5e-6, 5.0e-6])  #
    wlr_thmsk = np.array([7.6e-6, 8.9e-6])  #

    temperature_lookup_table_ircam, _, _ = calc_photons_to_temperature(temperatures, wavelength_range=wlr_ircam,
                                                                    emissivity=1, integration_time=1)
    temperature_lookup_table_flir, _, _ = calc_photons_to_temperature(temperatures, wavelength_range=wlr_flir,
                                                                    emissivity=1, integration_time=1)
    temperature_lookup_table_ircam_ndf, _, _ = calc_photons_to_temperature(temperatures, wavelength_range=wlr_ircam,
                                                                       emissivity=1, integration_time=1,
                                                                           transmittance=0.25)
    path_fn = '../input_files/mast_u/temperature_coefs-mast_u-2019_02_19.csv'
    data_calibration = read_csv(path_fn)
    # print(data_calibration)

    bg_photons_ircam_0101 = data_calibration.loc[0, 'photons_bg']
    bg_photons_ircam_0101_ndf = data_calibration.loc[1, 'photons_bg']
    bg_photons_ircam_0102 = data_calibration.loc[2, 'photons_bg']
    bg_photons_ircam_0102_ndf = data_calibration.loc[3, 'photons_bg']
    bg_photons_flir_45 = data_calibration.loc[4, 'photons_bg']
    bg_photons_flir_47 = data_calibration.loc[5, 'photons_bg']

    temperature_ircam_0101 = lookup_temperature_for_photon_count(temperature_lookup_table_ircam, bg_photons_ircam_0101)
    temperature_ircam_0102 = lookup_temperature_for_photon_count(temperature_lookup_table_ircam, bg_photons_ircam_0102)
    temperature_flir_45 = lookup_temperature_for_photon_count(temperature_lookup_table_flir, bg_photons_flir_45)
    temperature_flir_47 = lookup_temperature_for_photon_count(temperature_lookup_table_flir, bg_photons_flir_47)
    temperature_ircam_0101_ndf = lookup_temperature_for_photon_count(temperature_lookup_table_ircam,
                                                                     bg_photons_ircam_0101_ndf)
    temperature_ircam_0102_ndf = lookup_temperature_for_photon_count(temperature_lookup_table_ircam,
                                                                     bg_photons_ircam_0102_ndf)
    temperatures = dict(temperature_ircam_0101=temperature_ircam_0101-zero_Celsius,
                        temperature_ircam_0102=temperature_ircam_0102 - zero_Celsius,
                        temperature_flir_45=temperature_flir_45 - zero_Celsius,
                        temperature_flir_47=temperature_flir_47 - zero_Celsius,
                        temperature_ircam_0101_ndf=temperature_ircam_0101_ndf - zero_Celsius,
                        temperature_ircam_0102_ndf=temperature_ircam_0102_ndf - zero_Celsius,
                        )
    # NOTE: Why do IRCAM data woth NDF not need 25% transmittance reduction
    # print(pd.Series(temperatures))
    return temperatures

def compare_curve_to_old_lookup_file():
    """Work out what parameters were used to produce the old MAST scheduler BB curve lookup files
    Conclusions:
    - Good agreement for emissivity=1, solid_angle=2pi, integration_time=1s, wavelegth_step = 1.5e-8
    - Discrepancy is of order that resulting from numerical integration of photon flux. Setting
        wavelength_step=1.5e-8 causes discrepancy to be both sides of 0.
    - Smaller discrepancy if use 273 def offset for deg -> Kelvin rather than 273.15 (1.9% max discrepancy cf 2.5%)
    - With 273C offset and wl step of 1.5e-8 get max percentage error of: 0.52%
    """
    import matplotlib.pyplot as plt
    from fire.interfaces.interfaces import read_csv
    path_fn = '/home/tfarley/repos/air/fire/input_files/bb_photons-13mm-232us.tsv'
    bb_curve_legacy = read_csv(path_fn)  # , index_col='temperature_celcius')
    bb_curve_legacy['temperature'] = bb_curve_legacy['temperature_celcius'] + zero_Celsius
    bb_curve_legacy.set_index('temperature')

    bb_analytic, _, _ = calc_photons_to_temperature(bb_curve_legacy.index + 273,
                                                    wavelength_range=wlr_sbf,
                                                    emissivity=1, solid_angle=2*np.pi,
                                                    integration_time=1, wavelength_step=1.5e-8, transmittance=1)


    fig, (ax0, ax1) = plt.subplots(2, 1)
    ax0.plot(bb_curve_legacy['photon_flux'], bb_curve_legacy['temperature_celcius'], label='legacy')
    ax0.plot(bb_analytic['n_photons'], bb_analytic['temperature_celcius'], label='analytic')
    ax0.set_xlabel(r'Photons')
    ax0.set_ylabel(r'Temperature [C]')
    ax0.legend()

    photons_difference = bb_analytic['n_photons'] - bb_curve_legacy['photon_flux']
    diff_percent = (photons_difference / bb_curve_legacy['photon_flux'])*100
    ax1.plot(bb_curve_legacy['temperature_celcius'], bb_curve_legacy['photon_flux'],  label='legacy')
    ax1.plot(bb_analytic['temperature_celcius'], bb_analytic['n_photons'], label='analytic')
    ax1.plot(bb_analytic['temperature_celcius'], photons_difference,
            label='difference (new-leg)')
    ax1.set_xlabel(r'Temperature [C]')
    ax1.set_ylabel(r'Photons')
    # ax1.set_yscale('log')
    ax1.legend()

    print(f'Diff abs: {photons_difference}')
    print(f'Diff percent: {diff_percent}')
    print(f'Diff percent max: {np.max(np.abs(diff_percent))}')
    print(f'Diff percent av: {np.mean(diff_percent)}')

    plt.tight_layout()
    plt.show()
    pass

if __name__ == '__main__':
    compare_curve_to_old_lookup_file()
    data = calc_bg_temperatures_for_calibration()


    from fire.plotting.presentation_figures import plot_bb_curves
    plot_bb_curves()