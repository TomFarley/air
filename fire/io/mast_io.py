#!/usr/bin/env python

"""
Functions for loading and interacting with IR camera data from the MAST tokamak (2000-2013).


"""

import numpy as np
import xarray as xr

import pyIpx

def read_local_ipx_file(ipx_path: str) -> np.ndarray:
    frames = np.zeros((10,10))
    return frames