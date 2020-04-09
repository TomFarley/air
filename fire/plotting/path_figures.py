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
logger.setLevel(logging.DEBUG)


def figure_path(data, key, slice=None, ax=None, plot_kwargs=None):
    if slice is None:
        slice = {'n': np.median(data['n'])}
    kws = {'color': None}
    if isinstance(plot_kwargs, dict):
        kws.update(plot_kwargs)

    data_plot = data[key]
    if data_plot.ndim > 1:
        data_plot = data_plot.sel(slice)

    data_plot.plot.line(ax=ax, **kws)
    ax.title.set_fontsize(10)


if __name__ == '__main__':
    pass