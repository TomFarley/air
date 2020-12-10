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
logger.propagate = False



if __name__ == '__main__':
    # signals = 'all'
    # signals = ['AIR_RCOORD_ISP']
    signals = ['AIR_TPROFILE_OSP']
    uda_utils.plot_uda_signals_individually(23586, signals=signals, diagnostic_alias='air', min_rank=2)
    pass