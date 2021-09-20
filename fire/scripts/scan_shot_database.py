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

from fire.scripts.read_pickled_ir_data import read_data_for_pulses_pickle

logger = logging.getLogger(__name__)
logger.propagate = False

def load_analysed_shot_pickle(pulse, camera='rit', machine='mast_u', recompute=False):
    logger.info(f'Reviewing {machine}, {camera}, {pulse}')
    print(f'Reviewing {machine}, {camera}, {pulse}')

    data, data_unpacked = read_data_for_pulses_pickle(camera=camera, pulses=pulse, machine=machine,
                                                      generate=True, recompute=recompute)[pulse]

    image_data = data['image_data']
    path_data = data['path_data']

    meta = data['meta_data']
    # meta = dict(pulse=pulse, camera=camera, machine=machine)

def extract_signals_from_shot_range(shots, camera='rit', machine='mast_u', signals=(), recompute=False):
    signals = dict(signals)
    data_scan = pd.DataFrame(data=None, index=(), columns=list(signals.keys()))
    for shot in shots:
        data, data_unpacked = read_data_for_pulses_pickle(camera=camera, pulses=shot, machine=machine,
                                                          generate=True, recompute=recompute)[shot]
        for signal, func in signals.items():
            data_scan.loc[shot, signal] = func(data)

    return data_scan

def heat_flux_min(data):
    out = data['path_data']['heat_flux_path0'].min()
    return out

def heat_flux_max(data):
    out = data['path_data']['heat_flux_path0'].max()
    return out


if __name__ == '__main__':
    shot_start = 44963
    n_shots = 4
    shots = np.arange(shot_start, shot_start-(n_shots+1), -1)
    print(shots)

    signals = dict(heat_flux_min=heat_flux_min, heat_flux_max=heat_flux_max)

    data = extract_signals_from_shot_range(shots, signals=signals)

    pass