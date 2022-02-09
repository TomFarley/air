#!/usr/bin/env python

"""


Created: 
"""

import logging
from typing import Union, Iterable, Sequence, Tuple, Optional, Any, Dict
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# TODO: Transfer functionality from FEET repo

@dataclass
class EichFit:
    """Class working with Eich fits to divertor heatflux data"""
    q_0: float
    S: float
    lambda_q: float
    x_0: float
    q_bg: float
    x: np.ndarray = None

    def plot(self, ax=None, **kwargs):
        if 'label' not in kwargs:
            kwargs['label'] = self.label(params=('q_0', 'lambda_q', 'S'))

        line = plot_eich_fit(self.x, (self.q_0, self.S, self.lambda_q, self.x_0, self.q_bg), ax=ax, **kwargs)
        return line

    def label(self, params=('q_0', 'lambda_q', 'S')):
        param_syms = {'lambda_q': '\lambda_q',
                      # 'q_0': 'q_{\perp, 0}'
                      }
        units = {'q_0': '$ MW$', 'lambda_q': '$ mm$', 'S': '$ mm$'}
        factors = {'q_0': 1, 'lambda_q': 1e3, 'S': 1e3}
        parts = []
        for i, param in enumerate(params):
            value = getattr(self, param) * factors.get(param, 1)
            sym = param_syms.get(param, param)
            unit = units.get(param, '')
            parts.append(f'{sym}={value:0.3g}{unit}')

        parts_joined = ", ".join(parts)
        label = f'Eich(${parts_joined})'
        if label.count('$') // 2 > 0:
            label += '$'
        return label


def eich_profile_func(x, q_0, S, lambda_q, x_0, q_bg):
    from scipy.special import erfc

    return (q_0 / 2.0) * np.exp((S / (2.0 * lambda_q)) ** 2 - ((x - x_0) / lambda_q)) * erfc(
        (S / (2.0 * lambda_q)) - (x - x_0) / S) + q_bg

def eich_initial_guess(x, y):
    q0_upper = np.nanmax(y) * 5
    q0_lower = np.nanmax(y) * 0.8
    q0_guess = np.nanmax(y) * 2

    x0_guess = np.nanmean(x[y == np.nanmax(y)])
    x0_lower = np.nanmin(x) / 1.5
    x0_upper = np.nanmax(x) * 1.5

    x_gtr_x0 = x[x > x0_guess]
    y_gtr_x0 = y[x > x0_guess]
    lambda_q_guess = 0.8 * (x_gtr_x0[y_gtr_x0 < np.nanmax(y) / np.exp(1)][0] - x0_guess)
    lambda_q_upper = lambda_q_guess * 5
    lambda_q_lower = lambda_q_guess / 3

    S_guess = 0.4 * lambda_q_guess
    S_upper = S_guess * 3
    S_lower = S_guess / 3

    qbg_guess = 0
    qbg_upper = np.nanmax(y) / 15
    qbg_lower = -np.nanmax(y) / 15

    lower = [q0_lower, S_lower, lambda_q_lower, x0_lower, qbg_lower]
    upper = [q0_upper, S_upper, lambda_q_upper, x0_upper, qbg_upper]
    guess = [q0_guess, S_guess, lambda_q_guess, x0_guess, qbg_guess]

    return lower, upper, guess

def fit_eich_to_profile(x, y, y_err=None, eich_start_parameters=None):

    x_remove_xnan = x[~np.isnan(x)]
    y_remove_xnan = y[~np.isnan(x)]

    x_remove_xnan_ynan = x_remove_xnan[~np.isnan(y_remove_xnan)]
    y_remove_xnan_ynan = y_remove_xnan[~np.isnan(y_remove_xnan)]

    if y_err is not None:
        y_err_remove_xnan = y_err[~np.isnan(x)]
        y_err_remove_xnan_ynan = y_remove_xnan[~np.isnan(y_remove_xnan)]

    if eich_start_parameters is None:
        lower, upper, guess = eich_initial_guess(x_remove_xnan_ynan, y_remove_xnan_ynan)
    else:
        lower = [eich_start_parameters['q0_limit'][0], eich_start_parameters['S_limit'][0],
                 eich_start_parameters['lambda_q_limit'][0], eich_start_parameters['x0_limit'][0],
                 eich_start_parameters['qbg_limit'][0]]
        upper = [eich_start_parameters['q0_limit'][1], eich_start_parameters['S_limit'][1],
                 eich_start_parameters['lambda_q_limit'][1], eich_start_parameters['x0_limit'][1],
                 eich_start_parameters['qbg_limit'][1]]
        guess = [eich_start_parameters['q0'], eich_start_parameters['S'], eich_start_parameters['lambda_q'],
                 eich_start_parameters['x0'], eich_start_parameters['qbg']]


    try:
        # popt, pcov = curve_fit(self.eich_profile, x_remove_xnan_ynan, y_remove_xnan_ynan,sigma=y_err_remove_xnan_ynan,bounds=(lower,upper),p0=guess)
        popt, pcov = curve_fit(eich_profile_func, x_remove_xnan_ynan, y_remove_xnan_ynan, bounds=(lower, upper),
                               p0=guess)
    except Exception as e:
        raise e
    else:
        q0, S, lambda_q, x0, q_bg = popt

    return EichFit(*popt, x=x_remove_xnan_ynan)


def plot_eich_fit(x, popt, ax=None, **kwargs):
    from fire.plotting import plot_tools

    fig, ax, ax_passed = plot_tools.get_fig_ax(ax)

    fit_x_axis = np.linspace(np.nanmin(x), np.nanmax(x), 500)
    fit_y = eich_profile_func(fit_x_axis, *popt)

    if len(x) > 5:

        kws = dict(ls='--', color='previous', alpha=0.5, label='Eich()')
        kws.update(kwargs)

        if kws.get('color') == 'previous':
            kws['color'] = plot_tools.get_previous_line_color(ax)

        # plt.plot(fit_x_axis,self.eich_profile(fit_x_axis, *guess))
        line = ax.plot(fit_x_axis, fit_y, **kws)

        return line

if __name__ == '__main__':
    pass