# -*- coding: future_fstrings -*-
#!/usr/bin/env python

"""


Created: 
"""

import logging

import numpy as np

import fire.plotting.spatial_figures
import matplotlib.pyplot as plt

from fire.plotting import image_figures, spatial_figures, path_figures, plot_tools
from fire.plotting.image_figures import (figure_xarray_imshow, figure_frame_data, plot_outlier_pixels, figure_analysis_path,
                                         figure_spatial_res_max, figure_spatial_res_x, figure_spatial_res_y,
                                         plot_image_data_hist, plot_analysis_path, animate_image_data)
from fire.plotting.spatial_figures import figure_poloidal_cross_section, figure_top_down
from fire.plotting.path_figures import figure_path_1d, figure_path_2d
from fire.plotting.plot_tools import annotate_axis, repeat_color
from fire.camera.image_processing import find_outlier_pixels
from fire.misc.utils import make_iterable

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


def debug_movie_data(data, aspect='equal'):
    fig, axes = plt.subplots(2, 3, num='movie_data', figsize=(14, 8))
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

    ax = axes[5]
    image_figures.plot_image_data_temporal_stats(data, key='frame_data', ax=ax,
                                                 x_key='n', stats=('max', 'mean', 'median', 'min'))
    plt.tight_layout()
    plt.show()

def debug_detector_window(detector_window, frame_data=None, key='frame_data_nuc', calcam_calib=None,
                          image_full_frame=None, subview_mask_full_frame=None, aspect='equal', image_coords='Display'):

    if image_full_frame is None:
        # Get full frame calibration image without detector window applied
        calcam_calib.set_detector_window(None)
        image_full_frame = calcam_calib.get_image(coords=image_coords)
        subview_mask_full_frame = calcam_calib.get_subview_mask(coords=image_coords)
        calcam_calib.set_detector_window(detector_window)  # Re-apply detector_window

    fig_shape = (2, 3) if (frame_data is not None) else (1, 1)
    fig, axes = plt.subplots(*fig_shape, num='movie_data', figsize=(14, 8))

    if (frame_data is not None):
        axes = axes.flatten()
        ax = axes[0]
    else:
        ax = axes

    ax.imshow(image_full_frame, interpolation='none')

    # TODO: reinstate subview_mask when reseting detector_window is working
    # image_figures.plot_sub_view_masks(ax, calcam_calib, image_coords='Display')

    image_figures.plot_detector_window(ax, detector_window)

    if (frame_data is not None):
        # Frame data
        ax = axes[1]
        # image = frame_data[int(len(frame_data)/2)]
        # ax.imshow(image, cmap='gray', interpolation='none')

        figure_frame_data(frame_data, ax=ax, n='bright', key='frame_data_nuc', label_outliers=False, aspect=aspect, show=False)

        ax = axes[2]
        figure_frame_data(frame_data, ax=ax, n=None, key='frame_data_nuc', label_outliers=True, aspect=aspect, show=False)

        ax = axes[3]
        n = int(np.percentile(frame_data['n'], 75))
        figure_frame_data(frame_data, ax=ax, n=n, key='frame_data_nuc', label_outliers=False, aspect=aspect, show=False)

        ax = axes[4]
        n = int(np.percentile(frame_data['n'], 85))
        figure_frame_data(frame_data, ax=ax, n=n, key='frame_data_nuc', label_outliers=False, aspect=aspect, show=False)

        ax = axes[5]
        n = frame_data['n'].values[-1]
        figure_frame_data(frame_data, ax=ax, n=n, key='frame_data_nuc', label_outliers=False, aspect=aspect, show=False)

    plt.tight_layout()
    plt.show()

def debug_camera_shake(pixel_displacements, times=None, plot_float=True):
    fig, (ax1, ax2) = plt.subplots(2, 1, num='camera shake', figsize=(13, 13), sharex=True, sharey=True)

    pixel_displacements_int = np.round(pixel_displacements)
    # for i in [0, 1]:
    #     pixel_displacements_int[i, np.isnan(pixel_displacements[i])] = np.nan  # Avoid nans being cast to huge ints

    ax = ax1
    # x offset
    ax.plot(pixel_displacements_int[:, 0], label='x pixel offset', ls='-', alpha=0.8)
    if plot_float:
        color = repeat_color(ax, artist_string='-10')
        ax.plot(pixel_displacements[:, 0], label='x pixel offset (float)', ls='--', color=color, alpha=0.5)

    # y offset
    ax.plot(pixel_displacements_int[:, 1], label='x pixel offset', ls='-', alpha=0.8)
    if plot_float:
        color = repeat_color(ax, artist_string='-10')
        ax.plot(pixel_displacements[:, 1], label='y pixel offset (float)', ls='-.', color=color, alpha=0.5)

    ax = ax2
    # abs offset
    norm = np.linalg.norm(pixel_displacements, axis=1)
    norm_int = np.round(norm)
    ax.plot(norm_int, label='norm pixel offset', ls='-', alpha=0.8)
    if plot_float:
        color = repeat_color(ax, artist_string='-10')
        ax.plot(norm, label='norm pixel offset (float)', color=color, ls='-', alpha=0.5)

    ax1.set_xlabel('Frame index')
    ax1.set_ylabel('Camera shake [pixels]')
    ax1.legend()
    ax2.set_xlabel('Frame index')
    ax2.set_ylabel('Camera shake [pixels]')
    ax2.legend()

    plt.show()

def debug_spatial_coords(data, path_data=None, path_name='path0', points_rzphi=None, points_pix=None,
                         aspect='equal', axes_off=True):
    fig, axes = plt.subplots(3, 3, num='spatial coords', figsize=(13, 13), sharex=True, sharey=True)
    axes = axes.flatten()
    path = path_name

    if points_pix is not None:
        points_pix = np.array(points_pix)
        if points_pix.ndim == 1:
            points_pix = points_pix[np.newaxis, :]
        logger.info('Plotting %s user specified x_pix,y_pix points on image: %s', len(points_pix), points_pix)
    if points_rzphi is not None:
        points_rzphi = np.array(points_rzphi)
        if points_rzphi.ndim == 1:
            points_rzphi = points_rzphi[np.newaxis, :]
        logger.info('Plotting %s user specified r,z,phi points on image: %s', len(points_rzphi), points_rzphi)

    # Frame data
    ax = axes[0]
    figure_frame_data(data, ax=ax, key='frame_data_nuc', label_outliers=False, aspect=aspect, axes_off=False,
                      show=False)
    figure_xarray_imshow(data, key='subview_mask_im', ax=ax, alpha=0.3, add_colorbar=False, cmap='Pastel2', axes_off=False,
                         show=False)
    if (points_rzphi is not None):
        from fire import active_calcam_calib
        image_figures.plot_rzphi_points(active_calcam_calib, points_rzphi, ax=ax)
    if (points_pix is not None):
        ax.plot(points_pix[:, 0], points_pix[:, 1], **{'ls': '', 'marker': 'x', 'color': 'g'})

    if (path_data is not None) and (path_name is not None):
        plot_analysis_path(ax, path_data[f'x_pix_{path}'], path_data[f'y_pix_{path}'],
                           xpix_out_of_frame=path_data[f'x_pix_{path}_out_of_frame'],
                           ypix_out_of_frame=path_data[f'y_pix_{path}_out_of_frame'])
    # Spatial coords
    axs = axes[1:]
    keys = ['x_im', 'y_im', 'R_im', 'phi_deg_im', 'z_im', 's_global_im', 'sector_im', 'ray_lengths_im', 'wire_frame']
    for ax, key in zip(axs, keys):
        try:
            figure_xarray_imshow(data, key=key, ax=ax, axes_off=axes_off, show=False)
        except KeyError as e:
            logger.warning(f'Could not plot {str(e)}')
        if path_data is not None:
            path = path_name
            try:
                plot_analysis_path(ax, path_data[f'x_pix_{path}'], path_data[f'y_pix_{path}'],
                                   xpix_out_of_frame=path_data[f'x_pix_{path}_out_of_frame'],
                                   ypix_out_of_frame=path_data[f'y_pix_{path}_out_of_frame'])
            except Exception as e:
                raise
        if (points_rzphi is not None):
            from fire import active_calcam_calib
            image_figures.plot_rzphi_points(active_calcam_calib, points_rzphi, ax=ax)
        if (points_pix is not None):
            ax.plot(points_pix[:, 0], points_pix[:, 1], **{'ls': '', 'marker': 'x', 'color': 'g'})

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
    figure_xarray_imshow(data, key=key, ax=ax, show=False)

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
    figure_xarray_imshow(image_data, key=key, ax=ax, axes_off=True, show=False)

    ax = axes[2]
    key = 'material_id'
    figure_xarray_imshow(image_data, key=key, ax=ax, axes_off=True, show=False)

    plt.tight_layout()
    plt.show()

def debug_analysis_path_1d(image_data, path_data=None, path_names='path0', image_data_in_cross_sections=False,
                           machine_plugins=None):
    path_names = make_iterable(path_names)
    fig = plt.figure(constrained_layout=True, num='analysis paths', figsize=(10, 16))
    gs = fig.add_gridspec(ncols=3, nrows=5, width_ratios=[1, 1, 2])

    ax1 = fig.add_subplot(gs[:3, 0:2])
    figure_analysis_path(path_data, image_data, key='frame_data_nuc', path_names=path_names, ax=ax1, frame_border=True,
                         show=False, image_kwargs=dict(add_colorbar=False, axes_off=True))

    # Poloidal cross section of first wall
    ax2 = fig.add_subplot(gs[3:, 0])
    im_data = image_data if image_data_in_cross_sections else None
    try:

        fire.plotting.spatial_figures.figure_poloidal_cross_section(image_data=im_data, path_data=path_data, path_names=path_names,
                                                                    no_cal=True, legend=False, axes_off=True, ax=ax2, show=False)
    except Exception as e:
        pass  # Fails for MAST, JET due to no wall definition etc

    # Top down view
    ax3 = fig.add_subplot(gs[3:, 1])
    im_data = image_data if image_data_in_cross_sections else None
    figure_top_down(path_data=path_data, image_data=im_data, ax=ax3, path_names=path_names,
                    machine_plugins=machine_plugins, axes_off=True, phi_labels=False, legend=False, show=False)

    # Line plots of parameters along path
    if path_data:
        share_x = None
        keys = (('frame_data_{path}', 'frame_data_nuc_{path}'), 'temperature_{path}','s_global_{path}',
                'spatial_res_max_{path}',
                'surface_id_{path}')  # , 'sector_{path}'
        for i_row, keys_format in enumerate(keys):
            ax = fig.add_subplot(gs[i_row, 2], sharex=share_x)

            # plot_kwargs = dict(_labels=False)
            plot_kwargs = {}
            for key_format in make_iterable(keys_format):
                for path in path_names:
                    key = key_format.format(path=path)
                    figure_path_1d(path_data, key, ax=ax, plot_kwargs=plot_kwargs)

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
    plt.subplots_adjust(hspace=0.05)  # , wspace=0.1)
    plt.show()

def debug_analysis_path_2d(image_data, path_data=None, path_names='path0', image_data_in_cross_sections=False,
                           machine_plugins=None):

    path_names = make_iterable(path_names)
    fig = plt.figure(constrained_layout=True, num='analysis paths', figsize=(10, 16))
    gs = fig.add_gridspec(ncols=3, nrows=5, width_ratios=[1, 1, 2])

    ax1 = fig.add_subplot(gs[:3, 0:2])
    figure_analysis_path(path_data, image_data, key='frame_data_nuc', path_names=path_names, ax=ax1, frame_border=True,
                         show=False, image_kwargs=dict(add_colorbar=False, axes_off=True))

    # Poloidal cross section of first wall
    ax2 = fig.add_subplot(gs[3:, 0])
    im_data = image_data if image_data_in_cross_sections else None
    try:

        fire.plotting.spatial_figures.figure_poloidal_cross_section(image_data=im_data, path_data=path_data, path_names=path_names,
                                                                    no_cal=True, legend=False, axes_off=True, ax=ax2, show=False)
    except Exception as e:
        pass  # Fails for MAST, JET due to no wall definition etc

    # Top down view
    ax3 = fig.add_subplot(gs[3:, 1])
    im_data = image_data if image_data_in_cross_sections else None
    figure_top_down(path_data=path_data, image_data=im_data, ax=ax3, machine_plugins=machine_plugins,
                    axes_off=True, phi_labels=False, legend=False, show=False)

    # Line plots of parameters along path
    if path_data:
        share_x = None
        keys = ('frame_data_{path}', 'temperature_{path}','s_global_{path}', 'spatial_res_max_{path}',
                'surface_id_{path}')  # , 'sector_{path}'
        for i_row, key_format in enumerate(keys):
            ax = fig.add_subplot(gs[i_row, 2], sharex=share_x)

            # plot_kwargs = dict(_labels=False)
            plot_kwargs = {}
            for path in path_names:
                key = key_format.format(path=path)
                figure_path_2d(path_data, key, ax=ax, plot_kwargs=plot_kwargs)

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
    plt.subplots_adjust(hspace=0.05)  # , wspace=0.1)
    plt.show()



def debug_temperature_image(data):
    figure_xarray_imshow(data, key='temperature_im', show=True)

def debug_temperature_profile_2d(data_paths=None, path_names='path0'):
    for path_name in make_iterable(path_names):
        data = data_paths[f'temperature_{path_name}']
        data = data.swap_dims({'n': 't'})
        # data = data.swap_dims({f'i_{path_name}': f's_global_{path_name}'})
        data = data.swap_dims({f'i_{path_name}': f'R_{path_name}'})
        data.plot(robust=True, center=False, cmap='coolwarm')
        plt.show()


if __name__ == '__main__':
    pass