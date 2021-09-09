#!/usr/bin/env python

"""


Created: 
"""

import logging


import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from fire.plotting import plot_tools
from fire.plotting import temporal_figures
from fire.physics.physics_parameters import calc_2d_profile_param_stats
from fire.misc.data_structures import swap_xarray_dim

logger = logging.getLogger(__name__)
logger.propagate = False

def correct_dark_level_drift(frame_data, dark_level_correction_factors):
    dark_level_correction_factors = swap_xarray_dim(dark_level_correction_factors, frame_data.dims[0])
    name = frame_data.name
    frame_data = frame_data + dark_level_correction_factors
    frame_data.name = name
    frame_data.attrs['dark_level_correction_applied'] = True
    logger.info(f'Applied dark level drift correction with peak change of '
                f'{np.ptp(np.array(dark_level_correction_factors))}')
    return frame_data

def get_dark_level_drift(image_data, plot=False):
    image_data = swap_xarray_dim(image_data, 't')
    frame_data = image_data['frame_data']
    ray_lengths = image_data['ray_lengths_im']
    # ray_lengths = None

    mask_dark = get_dark_level_image_mask(frame_data, ray_lengths=ray_lengths, frame_range=[None, None])

    if plot:
        plot_dark_level_variation(frame_data, mask_dark)

    stat = 'mean'
    dark_level_stats, stat_keys = calc_2d_profile_param_stats(frame_data.where(mask_dark), stats=(stat,),
                                             coords_reduce=('y_pix', 'x_pix'),
                                         roll_width=51, roll_reduce_func='median', roll_center=False)
    dark_level = dark_level_stats[stat_keys[stat]]
    dark_level.name = 'dark_level'

    dark_level_correction = (int(dark_level[0]) - dark_level).astype(int)
    dark_level_correction.name = 'dark_level_correction'

    logger.info(f'Dark level variation is: {np.ptp(np.array(dark_level_correction))}')
    # temporal_figures.plot_temporal_stats(frame_data)
    return dark_level, dark_level_correction, mask_dark

def get_dark_level_image_mask(frame_data, ray_lengths=None, dark_percentile=10, std_percentile=10, ptp_percentile=15,
                              frame_range=[None, None], frac_dark=0.8, n_dark_pix_min=50):
    logger.info('Getting dark level mask')
    frame_data = np.array(frame_data)
    frame_data = frame_data[slice(*frame_range)]

    def get_dark_mask(frame_data, dark_percentile, frac_dark=0.8):
        mask_dark = (np.sum(frame_data <= np.percentile(frame_data, dark_percentile), axis=0)
                        > (frac_dark * len(frame_data)))
        return mask_dark

    # mask_dark = get_dark_mask(frame_data, dark_percentile, frac_dark=frac_dark)
    # while np.sum(mask_dark) < n_dark_pix_min:
    #     dark_percentile += 1
    #     mask_dark = get_dark_mask(frame_data, dark_percentile, frac_dark=frac_dark)

    std = np.std(frame_data, axis=0)
    mask_low_std = std <= np.percentile(std, std_percentile)

    ptp = np.ptp(frame_data, axis=0)
    mask_low_ptp = ptp <= np.percentile(ptp, ptp_percentile)

    mask_combined = mask_low_std * mask_low_ptp  # * mask_dark
    if ray_lengths is not None:
        mask_distance = ray_lengths <= 1.0  # Closest surfaces in MAST-U RIR view ~1.29 m
        mask_combined *= mask_distance

    n_dark = np.sum(mask_combined)
    if n_dark == 0:
        logger.warning('No dark areas of image identified for dark level drift check')

    return mask_combined

def plot_dark_level_variation(mask_dark, dark_level=None, frame_data=None):


    fig, axes, ax_passed = plot_tools.get_fig_ax(ax_grid_dims=(1, 2), figsize=(12, 8))

    ax = axes[0]
    ax.imshow(mask_dark)

    # mask = np.tile(mask, [len(frame_data), 1, 1])

    ax = axes[1]
    # temporal_figures.plot_temporal_stats(frame_data.where(mask), t=frame_data['t'], t_axis=0, ax=ax, stats=('mean'),
    #                                      show=False, ls=':')
    # temporal_figures.plot_temporal_stats(frame_data.where(mask), t=frame_data['t'], t_axis=0, ax=ax, stats=('mean'),
    #                                      roll_width=51, show=False, ls='-', roll_reduce_func='mean', roll_center=True)
    # temporal_figures.plot_temporal_stats(frame_data.where(mask), t=frame_data['t'], t_axis=0, ax=ax, stats=('mean'),
    #                                      roll_width=51, show=False, ls='-.', roll_reduce_func='mean', roll_center=False)

    if frame_data is not None:
        temporal_figures.plot_temporal_stats(frame_data.where(mask_dark), t=frame_data['t'], t_axis=0, ax=ax,
                                             stats=('mean'), roll_width=51, show=False, ls='--',
                                             roll_reduce_func='mean', roll_center=False)
    if dark_level is not None:
        (-dark_level).plot(ax=ax, label='Dark level')

    plot_tools.legend(ax)
    plot_tools.show_if(True)


if __name__ == '__main__':
    pass
