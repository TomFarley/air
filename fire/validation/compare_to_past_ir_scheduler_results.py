#!/usr/bin/env python

"""


Created: 
"""

import logging
from typing import Union, Iterable, Sequence, Tuple, Optional, Any, Dict
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from fire.interfaces import uda_utils

logger = logging.getLogger(__name__)




if __name__ == '__main__':
    # signals = 'all'
    # signals = ['AIR_RCOORD_ISP']
    # signals = ['AIR_TPROFILE_OSP']
    signals = ['AIR_QPROFILE_OSP']
    # pulse = 23586  # Full frame with clear spatial calibration
    # pulse = 26505  # Full frame OSP only louvre12d, 1D analysis profile, HIGH current - REQUIRES NEW CALCAM CALIBRATION
    pulse = 29541  # Full frame OSP only louvre12d, 1D analysis profile, HIGH current - REQUIRES NEW CALCAM CALIBRATION
    uda_utils.plot_uda_signals_individually(pulse, signals=signals, diagnostic_alias='air', min_rank=2, verbose=True)
    pass