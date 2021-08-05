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

from mast

logger = logging.getLogger(__name__)
logger.propagate = False

def lookup_b_field(gfile, ):
    equil = mastu_equilibrium()

    equil.load_efitpp('epm044427.nc', time=0.5)


if __name__ == '__main__':
    pass