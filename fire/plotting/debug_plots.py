# -*- coding: future_fstrings -*-
#!/usr/bin/env python

"""


Created: 
"""

import logging

import matplotlib.pyplot as plt

from fire.plotting.image_figures import (figure_imshow, figure_frame_data, plot_outlier_pixels, figure_analysis_path,
                                         figure_spatial_res_max, figure_spatial_res_x, figure_spatial_res_y, plot_image_data_hist)
from fire.plotting import image_figures
from fire.plotting.path_figures import figure_path
from fire.plotting.plot_tools import annotate_axis
from fire.camera.image_processing import find_outlier_pixels

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def debug_movie_data(data, aspect='equal'):
    fig, axes = plt.subplots(2, 3, num='movie_data', figsize=(13, 13))
    axes = axes.flatten()

    # Frame data
    ax = axes[0]
    figure_frame_data(data, ax=ax, n=0, label_outliers=True, aspect=aspect, show=False)

    ax = axes[1]
    figure_frame_data(data, ax=ax, n=None, label_outliers=False, aspect=aspect, show=False)

    ax = axes[2]
    n = data['n'].values[-1]
    figure_frame_data(data, ax=ax, n=n, label_outliers=False, aspect=aspect, show=False)

    ax = axes[3]
    figure_frame_data(data, key='nuc_frame', ax=ax, n=None, label_outliers=True, aspect=aspect, show=False)

    ax = axes[4]
    plot_image_data_hist(data, key='nuc_frame', xlabel='NUC frame DL', ax=ax, show=False)

    # TODO: Plot peak frame intensity variation

    plt.tight_layout()
    plt.show()

def debug_spatial_coords(data, aspect='equal', axes_off=True):
    fig, axes = plt.subplots(3, 3, num='spatial coords', figsize=(13, 13))
    axes = axes.flatten()

    # Frame data
    ax = axes[0]
    figure_frame_data(data, ax=ax, label_outliers=True, aspect=aspect, axes_off=False, show=False)

    # Spatial coords
    axs = axes[1:]
    keys = ['x_im', 'y_im', 'R_im', 'phi_deg_im', 'z_im', 's_global_im', 'sector_im', 'ray_lengths', 'wire_frame']
    for ax, key in zip(axs, keys):
        try:
            figure_imshow(data, key=key, ax=ax, axes_off=axes_off, show=False)
        except KeyError as e:
            logger.warning(f'Could not plot {str(e)}')

    plt.tight_layout()
    plt.show()

def debug_spatial_res(data, aspect='equal'):
    fig, axes = plt.subplots(2, 3, num='spatial resolution', figsize=(13, 13))
    axes = axes.flatten()

    # Frame data
    ax = axes[0]
    figure_frame_data(data, ax=ax, aspect=aspect, show=False)

    # Spatial coord
    ax = axes[1]
    # key = 'R_im'
    # key = 'phi_deg_im'
    # key = 'x_im'
    key = 'y_im'
    figure_imshow(data, key=key, ax=ax, show=False)

    # Spatial res hist
    ax = axes[2]
    plot_image_data_hist(data, key='spatial_res_max', ax=ax, xlabel='spatial res (max) [m]', log_x=True,
                         show=False, save_fn=None)

    # Spatial res
    ax = axes[3]
    figure_spatial_res_x(data, ax=ax, show=False, save_fn=None, aspect=aspect)
    ax = axes[4]
    figure_spatial_res_y(data, ax=ax, show=False, save_fn=None, aspect=aspect)
    ax = axes[5]
    figure_spatial_res_max(data, ax=ax, show=False, save_fn=None, aspect=aspect)

    plt.tight_layout()
    plt.show()

def debug_surfaces(image_data, aspect='equal'):
    ig, axes = plt.subplots(1, 3, num='surfaces', figsize=(13, 6))
    axes = axes.flatten()

    # Frame data
    ax = axes[0]
    figure_frame_data(image_data, ax=ax, aspect=aspect, show=False)

    # Surface id
    ax = axes[1]
    key = 'surface_id'
    figure_imshow(image_data, key=key, ax=ax, axes_off=True, show=False)

    ax = axes[2]
    key = 'material_id'
    figure_imshow(image_data, key=key, ax=ax, axes_off=True, show=False)

    plt.tight_layout()
    plt.show()

def debug_analysis_path(image_data, path_data=None, path_name='path0'):
    path = path_name
    fig = plt.figure(constrained_layout=True, num='analysis paths')
    gs = fig.add_gridspec(ncols=2, nrows=5, width_ratios=[1, 3])

    ax1 = fig.add_subplot(gs[:3, 0])
    figure_analysis_path(path_data, image_data, key='frame_data', path_name=path_name, ax=ax1, frame_border=True,
                         show=False, image_kwargs=dict(add_colorbar=False, axes_off=True))
    try:
        ax2 = fig.add_subplot(gs[3:, 0])
        image_figures.figure_poloidal_cross_section(image_data=None, path_data=path_data, path_name=path_name,
                                                    no_cal=True, legend=False, axes_off=True, ax=ax2, show=False)
    except Exception as e:
        pass # Fails for MAST

    # Line plots of parameters along path
    if path_data:
        share_x = None
        keys = (f'frame_data_{path}', f's_global_{path}', f'spatial_res_max_{path}', f'sector_{path}',
                f'surface_id_{path}')
        for i_row, key in enumerate(keys):
            ax = fig.add_subplot(gs[i_row, 1], sharex=share_x)

            # plot_kwargs = dict(_labels=False)
            plot_kwargs = {}
            figure_path(path_data, key, ax=ax, plot_kwargs=plot_kwargs)

            # Move y axis label to annotation
            title = ax.get_yaxis().label.get_text()
            ax.get_yaxis().label.set_visible(False)
            annotate_axis(ax, title, x=0.6, y=0.7, fontsize=10)

            ax.tick_params(axis='both', which='major', labelsize=10)

            if i_row != len(keys)-1:
                plt.setp(ax.get_xticklabels(), visible=False)
                ax.get_xaxis().label.set_visible(False)
            else:
                ax.get_xaxis().label.set_fontsize(10)

            if i_row == 0:
                share_x = ax

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)
    plt.show()

def debug_temperature(data):
    figure_imshow(data, key='temperature_im', show=True)


if __name__ == '__main__':
    pass