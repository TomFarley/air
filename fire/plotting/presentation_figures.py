#!/usr/bin/env python

"""


Created: 
"""

import logging
from typing import Union, Iterable, Sequence, Tuple, Optional, Any, Dict
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from fire.plotting.plot_tools import annotate_axis, legend, save_fig
from fire.physics.black_body import (planck_curve_wavelength_spectral_radiance, planck_curve_wavelength_photons,
    calc_wiens_displacment_law_wavelength)

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

def plot_bb_curves(show=True, equations=True):
    wlr_air = np.array([4.1e-6, 5.0e-6])
    wlr_ait = np.array([7.7e-6, 9.4e-6])
    wlr = wlr_air
    wl_step = 0.5e-8

    # wavelengths = np.arange(wlr[0], wlr[1], wl_step)
    wavelengths = np.arange(0, 30e-6, wl_step)

    temperature_range = [0, 501]
    temperatures = np.arange(*temperature_range, 100) + 273.15
    temperatures_peaks = np.arange(*temperature_range, 1) + 273.15

    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))
    radiance = planck_curve_wavelength_spectral_radiance(wavelengths, temperatures)
    photons = planck_curve_wavelength_photons(wavelengths, temperatures)

    # Plot peaks in rediance
    lamda_peaks = calc_wiens_displacment_law_wavelength(temperatures_peaks)
    radiance_peaks = [planck_curve_wavelength_spectral_radiance(lamda, temp)
                      for lamda, temp in zip(lamda_peaks, temperatures_peaks)]
    ax0.plot(lamda_peaks*1e6, radiance_peaks, ls='--', marker='', color='k', alpha=0.6)

    # Plot camera wavelegnth ranges
    [ax1.axvline(wl*1e6, ls='--', color='blue', alpha=0.8) for wl in wlr_air]
    [ax1.axvline(wl*1e6, ls='--', color='orange', alpha=0.8) for wl in wlr_ait]

    # Plot curves for different temperatures
    cmap = matplotlib.cm.get_cmap('jet')  # plasma
    colors = [cmap(i) for i in np.linspace(0.1, 0.9, len(temperatures))]
    for i, temp in reversed(list(enumerate(temperatures))):
        ax0.plot(wavelengths*1e6, radiance[:, i], label=f'$T$={temp-273.15:0.0f}$^\circ$C', color=colors[i])
        ax1.plot(wavelengths*1e6, photons[:, i], label=f'$T$={temp-273.15:0.0f}$^\circ$C', color=colors[i])

    if equations:
        annotate_axis(ax0, (r'$dI(\lambda, T)=  \frac{2hc^2}{\lambda^5}   \frac{1}{e^{ℎ𝑐/\lambda k T}−1} d\Omega d\lambda$'
                           '\n' + r'$[W \cdot sr^{-1} \cdot m^{-2} \cdot m^{-1}]$'),
                      0.92, 0.5, color='k', coords='axis', fontsize=20, horizontalalignment='right')
        annotate_axis(ax0, r'$\lambda_{max}=\epsilon \frac{b}{T}$ [m]', 0.5, 0.9, color=(0.4, 0.4, 0.4), coords='axis', fontsize=20,
                      horizontalalignment='center')
        annotate_axis(ax0, r'$j=\sigma T^4$ [W]', 0.5, 0.7, color='k', coords='axis', fontsize=20,
                      horizontalalignment='center')

        annotate_axis(ax1, (r'$\phi_{\gamma}(\lambda, T)=  \int_{\lambda_1}^{\lambda_2}\epsilon \frac{2\pi c}{'  # 
                            r'\lambda^4} \frac{1}{e^{ℎ𝑐/\lambda k T}−1} d\lambda$'
                           '\n' + r'$[s^{-1} \cdot m^{-2}]$'),
                      0.98, 0.5, color='k', coords='axis', fontsize=20, horizontalalignment='right')
        annotate_axis(ax1, 'LWIR\n(IRCAM)', np.mean(wlr_ait)*1e6, 5.5e28, color='orange', coords='data',
                      horizontalalignment='center', bbox=dict(alpha=0.8))
        annotate_axis(ax1, 'MWIR\n(FLIR)', np.mean(wlr_air)*1e6, 5.5e28, color='blue', coords='data',
                      horizontalalignment='center', bbox=dict(alpha=0.8))

    ax0.set_xlabel('$\lambda$ [$\mu$m]')
    ax0.set_ylabel(r'Radiance [$W sr^{-1} m^{-3}$]')
    ax0.set_xlim([0, None])
    ax0.set_ylim([0, None])
    legend(ax0)

    # ax1.set_xlabel('Temperature [C]')
    ax1.set_xlabel('$\lambda$ [$\mu$m]')
    ax1.set_ylabel(r'Photons [$s^{-1}m^{-3}$]')
    ax1.set_xlim([0, None])
    ax1.set_ylim([0, None])
    legend(ax1)

    plt.tight_layout()
    path_fn = '../figures/black_body_curves.png'
    save_fig(path_fn, fig=fig, verbose=True)
    if show:
        plt.show()
    return fig, (ax0, ax1)


if __name__ == '__main__':
    plot_bb_curves()
    pass