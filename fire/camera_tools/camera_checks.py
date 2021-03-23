#!/usr/bin/env python

"""


Created: 
"""

import logging
from typing import Union, Iterable, Sequence, Tuple, Optional, Any, Dict
from pathlib import Path

import numpy as np
from scipy import stats
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from fire.interfaces import uda_utils

logger = logging.getLogger(__name__)
logger.propagate = False

def get_camera_external_clock_info(camera, pulse):
    if camera.lower() in ['rit', 'ait']:
        signal_clock = 'xpx/clock/lwir-1'

    info_out = {}

    data = uda_utils.read_uda_signal_to_dataarray(signal_clock, pulse=pulse, raise_exceptions=False)

    dt_signal = 1e-4
    data = data.interp(t=np.arange(data['t'].min(), data['t'].max(), dt_signal))

    # dt_signal = stats.mode(data['t'].diff(dim='t')).mode
    t_high = data.sel(t=data > data.mean())['t']

    power = np.fft.fft(data)
    freq = np.fft.fftfreq(data.shape[-1], d=dt_signal)

    mask_pos = freq > 0
    power = power[mask_pos]
    freq = freq[mask_pos]

    power_abs = np.absolute(power)
    # plt.plot(freq, power.real, freq, power.imag)

    clock_freq = freq[np.argmax(power_abs)]  # TODO: Convert freq to int?

    info_out['clock_t_window'] = np.array([t_high[0], t_high[-1]])  # TODO: get start of last rising edge
    info_out['clock_frequency'] = clock_freq

    return info_out


if __name__ == '__main__':
    pass