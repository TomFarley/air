#!/usr/bin/env python

"""
Primary analysis workflow for MAST-U/JET IR scheduler analysis code.

Created: 10-10-2019
"""

from typing import Union, Iterable, Optional
from pathlib import Path

import numpy as np
import xarray as xr

def init_data_structures():
    """Return empty data structures for population by and use in fire code

    :return: settings, files, data, meta_data
    :rtype: dict, dict, xr.Dataset, dict
    """

    settings = {}
    files = {}
    data = xr.Dataset
    meta_data = {}

    return settings, files, data, meta_data