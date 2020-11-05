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

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

import fire.plugins.plugins as plugins
import fire.plugins.plugins_machine as plugins_machine
import fire.plugins.plugins_movie as plugins_movie
import fire.plugins.plugins_output_format as plugins_output_format

if __name__ == '__main__':
    pass