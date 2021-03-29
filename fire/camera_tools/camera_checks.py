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

from fire.misc import utils
from fire.interfaces import uda_utils

logger = logging.getLogger(__name__)
logger.propagate = False

def get_camera_external_clock_info(camera, pulse):
    if camera.lower() in ['rit', 'ait']:
        signal_clock = 'xpx/clock/lwir-1'

    info_out = {}

    try:
        data = uda_utils.read_uda_signal_to_dataarray(signal_clock, pulse=pulse, raise_exceptions=False)
    except Exception as e:
        return info_out
    data = data.astype(float)  # Switch from uint8

    t = data['t']

    signal_dv = np.concatenate([[0], np.diff(data)])

    frame_times = t[signal_dv > 1e-5]  # rising edges

    dt_frame = utils.mode_simple(np.diff(frame_times))  # time between frame aquisitions
    clock_freq = 1/dt_frame

    clock_peak_width = np.diff(t[np.abs(signal_dv) > 1e-5][:2])[0]  # Width of first square wave

    dt_signal = 1e-4
    data = data.interp(t=np.arange(t.min(), t.max(), dt_signal))

    # dt_signal = stats.mode(data['t'].diff(dim='t')).mode
    t_high = data.sel(t=data > data.mean())['t']

    power = np.fft.fft(data)
    freq = np.fft.fftfreq(data.shape[-1], d=dt_signal)

    mask_pos = freq > 0
    power = power[mask_pos]
    freq = freq[mask_pos]

    power_abs = np.absolute(power)
    # plt.plot(freq, power.real, freq, power.imag)

    clock_freq_fft = freq[np.argmax(power_abs)]  # TODO: Convert freq to int?

    info_out['clock_frame_times'] = frame_times  # rising edges
    info_out['clock_t_window'] = np.array([frame_times[0], frame_times[-1]])
    info_out['clock_nframes'] = len(frame_times)
    info_out['clock_frequency'] = clock_freq
    info_out['clock_inter_frame'] = dt_frame
    info_out['clock_square_wave_width'] = clock_peak_width
    info_out['clock_frequency_fft'] = clock_freq_fft

    return info_out

def get_frame_time_correction(frame_times_camera, frame_times_clock=None, clock_info=None,
                              singals_da=('xim/da/hm10/t', 'xim/da/hm10/r')):
    """Work out corrected frame times/frame rate

    Problem can occur when manually recording camera data, that the camera internal frame rate is configured
    differently to the externally supplied clock rate. In this situation the frame times drift out of sequence with
    the clock. Presumably the frames are aquired on the first internal clock cycle occuring after an trigger signal?
    Alternatively could look into only internal clock times that occur during clock high values (+5V top of square wave)

    Args:
        frame_times_camera:
        frame_times_clock:
        clock_info:
        singals_da:

    Returns:

    """
    time_correction = {}
    if frame_times_clock is not None:
        frame_times_corrected = np.full_like(frame_times_clock, np.nan)
        for i, t_clock in enumerate(np.array(frame_times_clock)):
            frame_times_corrected[i] = frame_times_camera[frame_times_camera >= t_clock][0]
        dt_mean = utils.mode_simple(np.diff(frame_times_corrected))
        fps_mean = 1/dt_mean
        fps_clock = 1/utils.mode_simple(np.diff(frame_times_clock))
        fps_camera = 1/utils.mode_simple(np.diff(frame_times_camera))
        factor = fps_mean / fps_camera
    time_correction = dict(factor=factor, frame_times_corrected=frame_times_corrected, fps_mean=fps_mean)
    return time_correction



if __name__ == '__main__':
    pass