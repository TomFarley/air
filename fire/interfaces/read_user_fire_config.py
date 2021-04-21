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

# To enable use in fire/__init__.py avoid any imports that import fire directly
from fire.interfaces import basic_io

logger = logging.getLogger(__name__)
logger.propagate = False


def read_user_fire_config(path_fn='~/.fire_config.json'):
    path_fn = Path(path_fn).expanduser()
    fire_config = basic_io.json_load(path_fn)
    return fire_config

if __name__ == '__main__':
    pass