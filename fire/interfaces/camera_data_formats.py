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

from fire.interfaces.interfaces import read_csv
from fire import fire_paths

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

fire_root = fire_paths['root']

def read_altair_asc_image_file(path_fn):
    """Read .asc image file produced by the Altair software.

    Args:
        path_fn: Path to file

    Returns: Dataframe of image data

    """
    image = read_csv(path_fn, sep='\t', skiprows=29, header=None,
                     # skipinitialspace=True,
                     # keep_default_na=False,
                     # na_filter=False,
                     # skip_blank_lines=True
                     )
    if np.all(np.isnan(image.iloc[:, -1])):
        # Remove erroneous column due to spaces at end of lines
        image = image.iloc[:, :-1]
    return image

def read_reasearchir_csv_image_file(path_fn):

    image = read_csv(path_fn, sep=',', skiprows=31, header=None)
    return image

def read_ircam_asc_image_file(path_fn, verbose=True):

    image = read_csv(path_fn, sep='\t', skiprows=0, header=None, verbose=verbose)
    return image

if __name__ == '__main__':
    path = Path(fire_root) / '../tests/test_data/lab/'

    bbname = 'bb_10us.ASC'
    bgname = 'bg_10us.ASC'
    image_bg = read_ircam_asc_image_file(path / bgname)
    image_bb = read_ircam_asc_image_file(path / bbname)
    image = image_bb - image_bg
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 5))
    ax0.imshow(image_bg, cmap='gray')
    ax1.imshow(image_bb, cmap='gray')
    ax2.imshow(image, cmap='gray')
    plt.show()

    bgname = 'background_100.asc'
    bbname = 'image_100.asc'
    image_bg = read_altair_asc_image_file(path/bgname)
    image_bb = read_altair_asc_image_file(path/bbname)
    image = image_bb - image_bg
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 5))
    ax0.imshow(image_bg, cmap='gray')
    ax1.imshow(image_bb, cmap='gray')
    ax2.imshow(image, cmap='gray')
    plt.show()

    bgname = 'test_bg_filter1.csv'
    bbname = 'test_image_filter1.csv'
    image_bg = read_reasearchir_csv_image_file(path/bgname)
    image_bb = read_reasearchir_csv_image_file(path/bbname)
    image = image_bb - image_bg
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 5))
    ax0.imshow(image_bg, cmap='gray')
    ax1.imshow(image_bb, cmap='gray')
    ax2.imshow(image, cmap='gray')
    plt.show()

    pass