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
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def format_poloidal_plane_ax(ax, units='m'):
    ax.set_xlabel(f'R [{units}]')
    ax.set_ylabel(f'Z [{units}]')
    ax.set_aspect('equal')
    return ax

def get_fig_ax(ax=None, num=None, fig_shape=(1, 1)):
    """If passed None for the ax keyword, return new figure and axes"""
    if ax is None:
        fig, ax = plt.subplots(*fig_shape, num=num, constrained_layout=True)
        ax_passed = False
    else:
        fig = ax.figure
        ax_passed = True
    return fig, ax, ax_passed

def annotate_axis(ax, string, x=0.85, y=0.955, fontsize=16,
                  bbox=(('facecolor', 'w'), ('ec', None), ('lw', 0), ('alpha', 0.5), ('boxstyle', 'round')),
                  horizontalalignment='center', verticalalignment='center', multialignment='left', **kwargs):
    if isinstance(bbox, (tuple, list)):
        bbox = dict(bbox)
    ax.text(x, y, string, fontsize=fontsize, bbox=bbox, horizontalalignment=horizontalalignment,
            verticalalignment=verticalalignment, transform=ax.transAxes, **kwargs)


if __name__ == '__main__':
    pass


def create_poloidal_cross_section_figure(nrow=1, ncol=1, cross_sec_axes=((0, 0),)):
    fig, axes = plt.subplots(nrow, ncol)
    if nrow==1 and ncol==1:
        format_poloidal_plane_ax(axes)
    else:
        for ax_coord in cross_sec_axes:
            ax = axes[slice(*ax_coord)]
            format_poloidal_plane_ax(ax)
    plt.tight_layout()
    return fig, axes