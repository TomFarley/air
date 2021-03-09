# -*- coding: future_fstrings -*-
#!/usr/bin/env python

"""


Created: 
"""

import logging
from pathlib import Path
from copy import copy, deepcopy

import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt
from cycler import cycler

from fire.plotting import image_figures, spatial_figures, path_figures, plot_tools, temporal_figures
from fire.plotting.image_figures import (figure_xarray_imshow, figure_frame_data, plot_outlier_pixels, figure_analysis_path,
                                         figure_spatial_res_max, figure_spatial_res_x, figure_spatial_res_y,
                                         plot_image_data_hist, plot_analysis_path, animate_image_data)
from fire.plotting.spatial_figures import figure_poloidal_cross_section, figure_top_down
from fire.plotting.path_figures import figure_path_1d, figure_path_2d
from fire.plotting.plot_tools import annotate_axis, repeat_color
from fire.camera.image_processing import find_outlier_pixels
from fire.plugins import plugins_machine
from fire.misc.utils import make_iterable
from fire.interfaces import uda_utils

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

colors = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red')

def debug_movie_data(data, key='frame_data', frame_nos=(0, None, -1), aspect='equal'):
    fig, axes = plt.subplots(2, 3, num='movie_data', figsize=(14, 8))
    axes = axes.flatten()

    frame_nos = [n if ((n is None) or (n > 0)) else data['n'].values[n] for n in frame_nos]  # sub negative n values

    # Frame data
    ax = axes[0]
    figure_frame_data(data, ax=ax, n=frame_nos[0], key=key, label_outliers=True, aspect=aspect, show=False)

    ax = axes[1]
    figure_frame_data(data, ax=ax, n=frame_nos[1], key=key, label_outliers=False, aspect=aspect, show=False)

    ax = axes[2]
    figure_frame_data(data, ax=ax, n=frame_nos[2], key=key, label_outliers=False, aspect=aspect, show=False)

    ax = axes[3]
    figure_frame_data(data, key='nuc_frame', ax=ax, n=None, label_outliers=True, aspect=aspect, show=False)

    ax = axes[4]
    plot_image_data_hist(data, key='nuc_frame', xlabel='NUC frame DL', ax=ax, show=False)

    ax = axes[5]
    image_figures.plot_image_data_temporal_stats(data, key=key, ax=ax,
                                                 x_key='n', stats=('max', 'mean', 'median', 'min'))
    plt.tight_layout()
    plt.show()

def debug_calcam_calib_image(calcam_calib, frame_data=None, frame_ref=None, n_frame_ref=None):
    title_size = 8

    calib_image_original = calcam_calib.get_image(coords='Original')
    calib_image_display = calcam_calib.get_image(coords='Display')

    if frame_data is not None:
        # TODO: Pick out bright frame instead of middle frame
        n = int(np.floor(len(frame_data) / 2))
        # n = frame_data.mean(dim=('x_pix', 'y_pix')).argmax(dim='n', skipna=True).values[0]
        # n = frame_data.mean(dim=('x_pix', 'y_pix')).argsort().values[-2]
        frame_display = frame_data[n]

    fig_shape = (2, 1+2*((calib_image_original is not None) and (frame_data is not None) or (frame_ref is not None)))
    fig, axes = plt.subplots(*fig_shape, num='calcam_calib_image', figsize=(14, 8), sharex='col', sharey='col')
    axes = axes.flatten()
    i_ax = 0

    if calib_image_original is not None:
        ax = axes[i_ax]
        ax.imshow(calib_image_original, interpolation='none', cmap='gray', origin='upper')
        ax.set_title(f'calcam_calib_image (Original): {Path(calcam_calib.filename).name}',
                     fontdict={'fontsize':title_size})
        i_ax += 1

    if frame_data is not None:
        ax = axes[i_ax]
        ax.imshow(frame_display, interpolation='none', cmap='gray', origin='upper')
        ax.set_title(f'movie image n={n}', fontdict={'fontsize': title_size})
        i_ax += 1

    if (frame_data is not None) and (calib_image_original is not None):
        i_ax += 0
        # Plot color combination of frame data and calibration image
        ax = axes[i_ax]
        n = int(np.floor(len(frame_data)/2))
        # colour_combined_image = np.zeros((3,)+frame_display.shape, dtype=int)
        colour_combined_image = np.zeros_like(calib_image_display)
        colour_combined_image[:, :, 0] = frame_display.values
        # colour_combined_image[:, :, 1] = frame_display.values
        colour_combined_image[:, :, 2] = calib_image_display[:, :, 0]
        ax.imshow(colour_combined_image, interpolation='none', origin='upper')
        ax.set_title(f'movie-calib superimposed', fontdict={'fontsize': title_size})
        i_ax += 1

    if frame_ref is not None:  # Typically movie frame before transformations
        ax = axes[i_ax]
        ax.imshow(frame_ref, interpolation='none', cmap='gray', origin='upper')
        ax.set_title(f'ref frame (pre-transforms) n={n_frame_ref}', fontdict={'fontsize': title_size})
        i_ax += 1

    if calib_image_original is not None:
        ax = axes[i_ax]
        ax.imshow(calib_image_display, interpolation='none', cmap='gray', origin='upper')
        ax.set_title(f'calcam_calib_image (Display): {Path(calcam_calib.filename).name}',
                     fontdict={'fontsize': title_size})
        i_ax += 1



    plt.tight_layout()
    plt.show()

def debug_detector_window(detector_window, frame_data=None, key='frame_data_nuc', calcam_calib=None,
                          image_full_frame=None, subview_mask_full_frame=None, aspect='equal',
                          image_coords='Display', image_full_frame_label=None):

    if image_full_frame is None:
        # Get full frame calibration image without detector window applied
        calcam_calib.set_detector_window(None)
        image_full_frame = calcam_calib.get_image(coords=image_coords)
        subview_mask_full_frame = calcam_calib.get_subview_mask(coords=image_coords)
        calcam_calib.set_detector_window(detector_window)  # Re-apply detector_window
        image_full_frame_label = 'Calcam calibration image (full frame)'
    if image_full_frame_label is None:
        image_full_frame_label = 'Full frame reference image'

    fig_shape = (2, 3) if (frame_data is not None) else (1, 1)
    fig, axes = plt.subplots(*fig_shape, num='detector sub window', figsize=(14, 8))

    if (frame_data is not None):
        axes = axes.flatten()
        ax = axes[0]
    else:
        ax = axes

    if image_full_frame is not None:
        ax.imshow(image_full_frame, interpolation='none')
        ax.set_title(image_full_frame_label, fontdict={'fontsize': 8})
    else:
        image_figures.plot_detector_window(ax, [0, 0] + list(calcam_calib.get_subview_mask().shape[::-1]))

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

def debug_camera_shake(pixel_displacements, times=None, plot_float=True, n_shake_ref=True):
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

    if n_shake_ref is not None:
        ax1.axvline(x=n_shake_ref, ls='--', color='k', label='Shake ref frame')
        ax2.axvline(x=n_shake_ref, ls='--', color='k', label='Shake ref frame')

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
    # cid = fig.canvas.mpl_connect('button_press_event', plot_tools.onclick)
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
                           xpix_out_of_frame=path_data[f'x_pix_out_of_frame_{path}'],
                           ypix_out_of_frame=path_data[f'y_pix_out_of_frame_{path}'])
    # Spatial coords
    axs = axes[1:]
    keys = ['x_im', 'y_im', 'R_im', 'phi_deg_im', 'z_im', 's_global_im', 'sector_im',
            # 'ray_lengths_im',
            'surface_id',
            # 'wire_frame',
            ]
    for ax, key in zip(axs, keys):
        try:
            figure_xarray_imshow(data, key=key, ax=ax, axes_off=axes_off, show=False)
        except KeyError as e:
            logger.warning(f'Could not plot {str(e)}')
        try:
            # plot_tools.CoordSelector(ax, data=data, vars=['R_im', 'phi_deg_im', 'z_im', 'x_im', 'y_im'])
            pass
        except Exception as e:
            raise e
        if path_data is not None:
            path = path_name
            try:
                plot_analysis_path(ax, path_data[f'x_pix_{path}'], path_data[f'y_pix_{path}'],
                                   xpix_out_of_frame=path_data[f'x_pix_out_of_frame_{path}'],
                                   ypix_out_of_frame=path_data[f'y_pix_out_of_frame_{path}'])
            except Exception as e:
                raise e
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
    # key = 'y_im'
    key = 'ray_lengths_im'
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
                           machine_plugins=None, pupil_coords=None,
                           keys_profiles=(
                                            # ('frame_data_{path}',
                                        ('frame_data_min(i)_{path}', 'frame_data_mean(i)_{path}',
                                        'frame_data_max(i)_{path}'),
                                        ('temperature_min(i)_{path}', 'temperature_mean(i)_{path}',
                                           'temperature_max(i)_{path}'),
                                          # 'heat_flux_r_peak_{path}',
                                          ('heat_flux_min(i)_{path}', 'heat_flux_mean(i)_{path}',
                                           'heat_flux_max(i)_{path}'),
                                          ('s_global_{path}', 'R_{path}'),
                                          'spatial_res_max_{path}',
                                          # 'sector_{path}'
                                          # 'surface_id_{path}'
                                          )  # ,
                           ):
    n_profiles = len(keys_profiles)
    path_names = make_iterable(path_names)
    fig = plt.figure(constrained_layout=True, num='analysis paths', figsize=(10, 12))
    gs = fig.add_gridspec(ncols=3, nrows=n_profiles, width_ratios=[1, 1, 3])

    i_ax_split = int(np.ceil(n_profiles/2 + 0.5))

    ax1 = fig.add_subplot(gs[:i_ax_split, 0:2])
    figure_analysis_path(path_data, image_data, key='frame_data_nuc', path_names=path_names, ax=ax1, frame_border=True,
                         show=False, image_kwargs=dict(add_colorbar=False, axes_off=True))

    # Poloidal cross section of first wall
    ax2 = fig.add_subplot(gs[i_ax_split:, 0])
    im_data = image_data if image_data_in_cross_sections else None
    try:

        spatial_figures.figure_poloidal_cross_section(image_data=im_data, path_data=path_data, path_names=path_names,
                pupil_coords=pupil_coords, no_cal=True, legend=False, axes_off=True, ax=ax2, show=False)
    except Exception as e:
        pass  # Fails for MAST, JET due to no wall definition etc

    # Top down view
    ax3 = fig.add_subplot(gs[i_ax_split:, 1])
    im_data = image_data if image_data_in_cross_sections else None
    figure_top_down(path_data=path_data, image_data=im_data, ax=ax3, path_names=path_names,
                    machine_plugins=machine_plugins, pupil_coords=pupil_coords, axes_off=True, phi_labels=False,
                    legend=False, show=False)

    # Line plots of parameters along path
    if path_data:
        share_x = None
        for i_row, keys_format in enumerate(keys_profiles):
            ax = fig.add_subplot(gs[i_row, 2], sharex=share_x)

            # plot_kwargs = dict(_labels=False)
            nlines = len(make_iterable(keys_format))
            plot_kwargs = dict(alpha=0.7)
            for key_format in make_iterable(keys_format):
                for path in path_names:
                    key = key_format.format(path=path)
                    # figure_path_1d(path_data, key, ax=ax, plot_kwargs=plot_kwargs)
                    spatial_figures.figure_spatial_profile_1d(path_data, key, path_name=path, ax=ax,
                                                              plot_kwargs=plot_kwargs, legend=(nlines > 1), show=False)

            # Move y axis label to annotation
            ax.get_yaxis().label.set_visible(False)
            if nlines == 1:
                title = ax.get_yaxis().label.get_text()
                annotate_axis(ax, title, x=0.99, y=0.98, fontsize=10,
                              horizontalalignment='right', verticalalignment='top')
            # else:
            #     plt.legend()

            ax.tick_params(axis='both', which='major', labelsize=10)

            if i_row != len(keys_profiles)-1:
                plt.setp(ax.get_xticklabels(), visible=False)
                ax.get_xaxis().label.set_visible(False)
            else:
                ax.get_xaxis().label.set_fontsize(10)

            if i_row == 0:
                share_x = ax
            elif i_row == n_profiles-1:
                ax.set_xlabel('Path index')
                # ax.set_xlabel(path_data.coords)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)  # , wspace=0.1)
    plt.show()

def debug_analysis_path_2d(image_data, path_data=None, path_names='path0', image_data_in_cross_sections=False,
                           machine_plugins=None,
                           keys_profiles=('frame_data_{path}', 'temperature_{path}', 's_global_{path}',
                                           'spatial_res_max_{path}')):  #, 'surface_id_{path} , 'sector_{path}'):

    path_names = make_iterable(path_names)
    fig = plt.figure(constrained_layout=True, num='analysis paths', figsize=(10, 12))
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
        for i_row, key_format in enumerate(keys_profiles):
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

            if i_row != len(keys_profiles)-1:
                plt.setp(ax.get_xticklabels(), visible=False)
                ax.get_xaxis().label.set_visible(False)
            else:
                ax.get_xaxis().label.set_fontsize(10)

            if i_row == 0:
                share_x = ax

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05)  # , wspace=0.1)
    plt.show()

def debug_analysis_path_cross_sections(path_data=None, image_data=None, path_names='path0',
                                       image_data_in_cross_sections=False, machine_plugins=None, pupil_coords=None,
                                       show=True, path_fn_save=None):
    path_names = make_iterable(path_names)

    fig, axes, ax_passed = plot_tools.get_fig_ax(ax_grid_dims=(1, 2))

    im_data = image_data if image_data_in_cross_sections else None

    # Poloidal cross section of first wall
    ax = axes[0]
    try:

        spatial_figures.figure_poloidal_cross_section(image_data=im_data, path_data=path_data, path_names=path_names,
                                                      pupil_coords=pupil_coords, no_cal=True, legend=False,
                                                      axes_off=True, ax=ax, show=False)
    except Exception as e:
        pass  # Fails for MAST, JET due to no wall definition etc

    # Top down view
    ax = axes[1]
    im_data = image_data if image_data_in_cross_sections else None
    figure_top_down(path_data=path_data, image_data=im_data, ax=ax, path_names=path_names,
                    machine_plugins=machine_plugins, pupil_coords=pupil_coords, axes_off=True, phi_labels=False,
                    legend=False, show=False)

    plot_tools.show_if(show)
    plot_tools.save_fig(path_fn_save)
    return fig


def debug_temperature_image(data):
    figure_xarray_imshow(data, key='temperature_im', show=True)

def debug_plot_profile_2d(data_paths, param='temperature', path_names='path0', robust=True, extend='min',
                          annotate=True, mark_peak=True, meta=None, ax=None, t_range=(0.0, 0.6), t_wins=None,
                          machine_plugins=None,
                          colorbar_kwargs=None, show=True,
                          verbose=True):
    # TODO: Move general code to plot_tools.py func

    for i_path, path_name in enumerate(make_iterable(path_names)):
        if ax is None:
            fig, ax_i = plt.subplots(1, 1, num=f'{param}_profile_2d {path_name}')
        else:
            ax_i = make_iterable(ax)[i_path]

        data = data_paths[f'{param}_{path_name}']
        if 'n' in data.dims:
            data = data.swap_dims({'n': 't'})
        # data = data.swap_dims({f'i_{path_name}': f's_global_{path_name}'})

        colorbar_kwargs = colorbar_kwargs if (colorbar_kwargs is not None) else {}
        cmap = cm.get_cmap('coolwarm')
        kws = plot_tools.setup_xarray_colorbar_ax(ax_i, data_plot=data, add_colorbar=True, robust=robust, extend=extend,
                                                  cmap=cmap, **colorbar_kwargs)

        coord_path = f'R_{path_name}'
        try:
            if f'i_{path_name}' in data.dims:
                data = data.swap_dims({f'i_{path_name}': coord_path})
            # Remove any nans from path coord as will make 2d plot fail
            mask_coord_nan = np.isnan(data[coord_path])
            if np.any(mask_coord_nan):
                data = data.sel({coord_path: ~mask_coord_nan})
            try:
                artist = data.plot(center=False, ax=ax_i, **kws)  # pcolormesh
            except ValueError as e:
                # contourf can handle irregular x axis
                artist = data.plot.contourf(levels=200, center=False, ax=ax_i, **kws)


        except (KeyError, ValueError) as e:
            # Value error due to: R data not monotonic or nans in coord values - switch back to index or s_path

            # data = data.sortby('')
            # data = data.swap_dims({f'R_{path_name}': f's_path_{path_name}'})
            if f'R_{path_name}' in data.dims:
                data = data.swap_dims({f'R_{path_name}': f'i_{path_name}'})

            artist = data.plot(center=False, ax=ax_i, **kws)

        param_peak = f'{param}_peak_{path_name}'
        param_r_peak = f'{param}_r_peak_{path_name}'
        if mark_peak and (param_r_peak in data_paths):
            data_peak = data_paths[param_peak].values
            data_r_peak = data_paths[param_r_peak]
            mask_low = data_peak < (np.nanmin(data_peak) + 0.01 * (np.nanmax(data_peak)-np.nanmin(data_peak)))
            # TODO: filter instead with moving window stdev of r pos?
            data_r_peak.loc[mask_low] = np.nan
            ax_i.plot(data_r_peak.values, data_r_peak[data.dims[0]], ls=':', color='g', lw=1.5, alpha=0.6, label='peak')

        if annotate and (meta is not None):
            # TODO: Make figure pulse label func
            # plot_tools.annotate_providence(ax_i, loc='top_right', meta=meta, box=False)
            plot_tools.annotate_providence(ax_i, loc='top left', meta_data=meta, box=False)

            # label = f'{meta["machine"]} {meta["camera"]} {meta["pulse"]}'.replace('_u', '-U').upper()
            # annotate_axis(ax_i, label, x=0.99, y=0.99, fontsize=12, box=False, horizontalalignment='right',
            #                                                                             verticalalignment='top')

        if t_range is not None:
            t_range = np.array(t_range)
            t_range[1] = np.min([t_range[1], data['t'].values.max()])
            ax_i.set_ylim(t_range)

        if machine_plugins is not None:
            if isinstance(machine_plugins, str):
                machine_plugins = plugins_machine.get_machine_plugins(machine=machine_plugins)
            try:
                machine_plugins['label_tiles'](ax_i, data.coords, coords_axes=data.dims, y=t_range[0])
            except Exception as e:
                logger.warning(f'Failed to call machine plugin to label tiles: {e}')

        if t_wins is not None:  # Add horizontal lines labeling times
            plot_tools.label_axis_windows(windows=t_wins, labels=t_wins, ax=ax_i, axis='y', line_kwargs=None)

        if extend == 'min':
            cmap.set_over(None)

        # TODO: Switch to using same routine as for uda_utils.plot_uda_dataarray
        if verbose:
            logger.info(f'{param}({path_name}): min={data.min().values:0.4g}, mean={data.mean().values:0.4g}, '
                        f'99%={np.percentile(data.values,99):0.4g}, max={data.max().values:0.4g}')

        plot_tools.show_if(show=show, tight_layout=True)
    return ax, data, artist

def debug_plot_spatial_profile_1d(data_paths, param='temperature', path_names='path0', t=None):
    raise NotImplementedError
    for path_name in make_iterable(path_names):
        # TODO: Move general code to plot_tools.py func
        plt.figure(f'{param}_profile_2d {path_name}')
        data = data_paths[f'{param}_{path_name}']
        if 'n' in data.dims:
            data = data.swap_dims({'n': 't'})
        if t is None:
            t_slices = identify_profile_time_highlights()
        else:
            t_slices = t

        # data = data.swap_dims({f'i_{path_name}': f's_global_{path_name}'})
        data = data.swap_dims({f'i_{path_name}': f'R_{path_name}'})
        try:
            data.plot(robust=True, center=False, cmap='coolwarm')
        except ValueError as e:
            # R data not monotonic - switch back to index or s_path
            # data = data.sortby('')
            # data = data.swap_dims({f'R_{path_name}': f's_path_{path_name}'})
            data = data.swap_dims({f'R_{path_name}': f'i_{path_name}'})
            data.plot(robust=True, center=False, cmap='coolwarm')
        plt.tight_layout()
        plt.show()

def debug_plot_temporal_profile_1d(data_paths, params=('heat_flux_r_peak', 'heat_flux_peak'), path_names='path0',
                                   x_var='t', heat_flux_thresh=-0.0, meta_data=None):
    # TODO: Move general code to plot_tools.py func
    colors = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red')
    for path in make_iterable(path_names):
        fig, ax1, ax_passed = plot_tools.get_fig_ax(num=f'{params} temporal profile {path}')
        ax = ax1
        ax.tick_params(axis='y', labelcolor=colors[0])

        if heat_flux_thresh not in (None, False):
            peak_heat_flux = data_paths[f'heat_flux_peak_{path}']
            mask_pos_heat_flux = peak_heat_flux > heat_flux_thresh
            peak_heat_flux_pos = peak_heat_flux[mask_pos_heat_flux]

        for i, (param, color) in enumerate(zip(make_iterable(params), colors)):
            if i == 1:
                ax = ax1.twinx()
                ax.tick_params(axis='y', labelcolor=color)
            key = f'{param}_{path}'
            data = data_paths[key]
            if x_var not in data.dims:
                data = data.swap_dims({data.dims[0]: x_var})

            data_pos_q = deepcopy(data)
            data_pos_q[~mask_pos_heat_flux] = np.nan

            try:
                data.plot(label=f'{param} (all)', ax=ax, ls=':', alpha=0.3, color=color)
                fire.plotting.temporal_figures.plot_temporal_profile(data_paths, key, path_name=path, mask=mask_pos_heat_flux,
                                                                     label=f'{param} (pos q)', ax=ax, ls='-', alpha=0.6, color=color, show=False)
            except ValueError as e:
                raise NotImplementedError
                # R data not monotonic - switch back to index or s_path
                # data = data.sortby('')
                # data = data.swap_dims({f'R_{path_name}': f's_path_{path_name}'})
                data = data.swap_dims({f'R_{path}': f'i_{path}'})
                data.plot(robust=True, center=False, cmap='coolwarm')
            else:
                plot_tools.legend(ax=ax)
        if heat_flux_thresh not in (None, False):
            ax.axhline(heat_flux_thresh, ls='--', color='k')
            ax.set_xlim([peak_heat_flux_pos['t'].min(), peak_heat_flux_pos['t'].max()])

        plot_tools.annotate_providence(ax, meta_data=meta_data)

        plot_tools.show_if(show=True, tight_layout=True)

    return fig, ax

def debug_plot_timings(data_profiles, pulse, params=('heat_flux_peak_{path}','temperature_peak_{path}',),
                       path_name='path0',
                       comparison_signals=(('xim/da/hm10/t', 'xim/da/hm10/r'), 'xpx/clock/lwir-1'), separate_axes=True):
    from fire.interfaces import uda_utils
    from fire.physics.physics_parameters import find_peaks_info
    # client = uda_utils.get_uda_client(use_mast_client=True, try_alternative=True)

    n_peaks_label = 2
    figsize = (10, 10)
    ylabel_size = 12

    if not separate_axes:
        n_axes = 1
        fig, axes, ax_passed = plot_tools.get_fig_ax(num='timings check', figsize=figsize, ax_grid_dims=(1, 1))
    else:
        n_axes = len(params) + len(comparison_signals)
        fig, axes, ax_passed = plot_tools.get_fig_ax(num='timings check', figsize=figsize, ax_grid_dims=(n_axes, 1),
                                                                                               sharex=True)#'col')
    axes = make_iterable(axes, cast_to=np.ndarray).flatten()
    i_ax = 0

    t_peaks = []

    for signals in make_iterable(comparison_signals):
        ax = axes[i_ax]
        signals = make_iterable(signals)
        n_sigs = len(signals)
        for i_sig, sig in enumerate(signals):
            data = uda_utils.read_uda_signal_to_dataarray(sig, pulse=pulse)
            peaks_info = find_peaks_info(data)
            t_peaks.append(peaks_info['x_peaks'].values[:n_peaks_label])  # Record time of 3 largest peaks

            normalise = False
            alpha = 0.6 if n_sigs > 1 else 1
            color = None
            if (n_sigs == 2):
                color = colors[i_sig]
                if (i_sig == 0):
                    ax.yaxis.label.set_size(ylabel_size)
                    ax.tick_params(axis='y', labelcolor=color)
                    plot_tools.legend(ax, only_multiple_artists=False)
                if (i_sig == 1):
                    # 2nd y axis for two signals
                    ax = ax.twinx()
                    ax.tick_params(axis='y', labelcolor=color)
                    plot_tools.legend(ax, only_multiple_artists=False)
                    # TODO: Make combined legend for shared axes
            elif n_sigs > 2 and i_sig > 0:
                normalise = True

            uda_utils.plot_uda_signal(sig, pulse, show=False, ax=ax, marker='', label=sig, color=color, alpha=alpha,
                                      normalise=normalise)
            plot_tools.legend(ax)

        if separate_axes:
            ax.yaxis.label.set_size(ylabel_size)
            i_ax += 1

    for param in make_iterable(params):
        key = param.format(path=path_name)
        data = data_profiles[key]
        peaks_info = find_peaks_info(data)
        t_peaks.append(peaks_info['x_peaks'].values[:n_peaks_label])  # Record time of 3 largest peaks

        ax = axes[i_ax]
        # param.format(path=path_name)
        fire.plotting.temporal_figures.plot_temporal_profile(data_profiles, param=param, path_name=path_name, ax=ax,
                                                             show=False, label=True)
        plot_tools.legend(ax)

        if separate_axes:
            ax.yaxis.label.set_size(ylabel_size)
            ax.get_xaxis().set_visible(False)
            i_ax += 1
    ax.get_xaxis().set_visible(True)  # make final x axis visible

    props_cycle = (cycler(ls=['--', '-.', ':', (0, (2, 5)), (0, (2, 9))]))
    for peaks, props in zip(t_peaks, props_cycle):
        for ax in axes:
            [ax.axvline(peak, alpha=0.5, color='k', **props) for peak in peaks]

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.03)
    plt.show()

def plot_mixed_fire_uda_signal(signal, path_data=None, meta_data=None, ax=None, plot_kwargs=None,
                               label='{pulse}', **kwargs):
    if meta_data is None:
        meta_data = {}

    pulse = meta_data.get('pulse', None)

    meta_data.setdefault('signal', signal)  # TODO: move outside func/apply to copy?
    if label is not None:
        label = label.format(**meta_data)

    success = False
    if path_data is not None:
        # FIRE target profile
        if not success:
            try:
                fig, ax, data_plot, artist = spatial_figures.figure_spatial_profile_1d(path_data, key=signal, ax=ax,
                                                        label=label, plot_kwargs=plot_kwargs, show=False, **kwargs)
                # path_figures
            except Exception as e:
                # raise
                pass
            else:
                success = True
                return data_plot, artist

    if path_data is not None:
        if not success:
            try:
                ax, data_plot, artist = debug_plot_profile_2d(path_data, param=signal, ax=ax, show=False, **kwargs)
                # path_figures
            except Exception as e:
                pass
            else:
                success = True
                return data_plot, artist

    if (not success) and (pulse is not None):
        try:
            ax, data_plot, artist = uda_utils.plot_uda_signal(signal, pulse=meta_data['pulse'], ax=ax, label=label,
                                                              show=False, **plot_kwargs)
        except Exception as e:
            raise
            pass
        else:
            success = True
            return data_plot, artist

    if not success:
        raise ValueError(f'Failed to plot signal "{signal}"')

def plot_mixed_fire_uda_signals(signals, pulses=None, path_data=None, meta_data=None, axes=None, sharex=None,
                                sharey='row', x_range=(-0.05, 0.6), separate_pulse_axes=True,
                                normalise_sigs_on_same_ax=True):
    """
    signals (n_axes,
                (y_axis1,
                    (normalised signals)
                y_axis2,
                    (normalised signals)
                )
            )
    Args:
        signals:
        path_data:
        meta_data:
        axes:

    Returns:

    """
    line_styles = (cycler(ls=['-', '--', '-.', ':', (0, (2, 5)), (0, (2, 9))]))

    if pulses is None:
        pulses = make_iterable(meta_data.get('pulse'))
    n_pulses = len(pulses)

    if meta_data is None:
        meta_data = {}

    n_ax_rows = len(signals)
    if separate_pulse_axes:
        n_ax_cols = n_pulses
        label = '{signal}'  # format string
        annotate_format = '{machine} {camera} {pulse} {path_label}'
    else:
        n_ax_cols = 1
        label = '{pulse}'  # format string
        annotate_format = '{machine} {camera}'

    n_ax_grid = (n_ax_rows, n_ax_cols)

    fig, axes, ax_passed = plot_tools.get_fig_ax(ax=axes, ax_grid_dims=n_ax_grid, sharex=sharex, sharey=sharey,
                                                 figsize=(14, 8))
    axes = np.array(make_iterable(axes))
    if n_ax_cols == 1:
        axes = axes[..., np.newaxis]

    for i_pulse, pulse in enumerate(pulses):
        # First loop over pulse/ax columns
        meta_data['pulse'] = pulse
        if separate_pulse_axes:
            i_ax_col = i_pulse
        else:
            i_ax_col = 0

        if (path_data is None) or (n_pulses > 1):  # TODO: Allow passing pulse keyed dict of pathdata?
            from fire.scripts.read_pickled_ir_data import read_data_for_pulses_pickle
            try:
                camera = meta_data['camera']
                machine = meta_data['machine']
                data = read_data_for_pulses_pickle(camera, pulse, machine, recompute=False)
                path_data = data[pulse][0]['path_data']
            except (FileNotFoundError, KeyError) as e:
                pass

        for i_ax_row, signals_ax in enumerate(signals):
            signals_ax = make_iterable(signals_ax)
            ax_left = axes[i_ax_row, i_ax_col]
            ax = ax_left
            ax_right = None

            if i_ax_row == 0:
                # Only annotate top axis
                plot_tools.annotate_providence(ax, label=annotate_format, meta_data=meta_data, loc='top left')

            if (len(signals_ax) == 2) and (isinstance(signals_ax[0], (list, tuple, np.ndarray))):
                ax_right = ax_left.twinx()
                ax_left.set_zorder(ax_right.get_zorder() + 1)  # Put first axis on top
                ax_left.patch.set_visible(False)

                axes_lr = (ax_left, ax_right)

                signals_left, signals_right = make_iterable(signals_ax[0]), make_iterable(signals_ax[1])

                for signals_side, color, ax in reversed(list(zip((signals_left, signals_right), colors, axes_lr))):
                    signals_side = make_iterable(signals_side)
                    n_signals_side = len(signals_side)
                    ax.tick_params(axis='y', labelcolor=color)

                    if separate_pulse_axes or (n_pulses == 1 and n_signals_side > 1):
                        label = '{signal}'
                    else:
                        if (n_pulses > 1) and (n_signals_side > 1):
                            label = '{pulse} {signal}'
                        else:
                            label = '{pulse}'

                    alpha = 1 if (len(signals_side) == 1) and (n_pulses == 1 or separate_pulse_axes) else 0.7
                    normalise_factor = False

                    for i, (signal, ls) in enumerate(zip(signals_side, line_styles)):
                        meta_data['signal'] = signal

                        data_plot, artist = plot_mixed_fire_uda_signal(signal, path_data=path_data, meta_data=meta_data,
                                                        ax=ax, normalise_factor=normalise_factor,
                                                    label=label, plot_kwargs=dict(color=color, alpha=alpha, **ls),
                                                                       legend=False)
                        normalise_factor = np.nanmax(data_plot)  # For subsequent signals on same side axis, normalise
                        # TODO: Update legend to give normalisation
                if separate_pulse_axes:
                    if (len(signals_side) > 1) and (i_ax_col == n_ax_cols-1):
                        plot_tools.legend(ax=ax, only_multiple_artists=False, box=True)
                else:
                    if n_pulses > 0:
                        plot_tools.legend(ax=ax_left, only_multiple_artists=False, box=True)
            else:
                # Plot all signals to left axis
                signals_side = make_iterable(signals_ax)

                if (len(signals_side) == 1) and isinstance(signals_side[0], (list, tuple, np.ndarray)):
                    #  single tuple of signals means they should all be plotted on left axis
                    signals_side = signals_side[0]
                n_signals_side = len(signals_side)

                if separate_pulse_axes or (n_pulses == 1 and n_signals_side > 1):
                    label = '{signal}'
                else:
                    if (n_pulses > 1) and (n_signals_side > 1):
                        label = '{pulse} {signal}'
                    else:
                        label = '{pulse}'
                alpha = 1 if (len(signals_side) == 1) and (n_pulses == 1 or separate_pulse_axes) else 0.7
                normalise_factor = False
                for i_left, signal in enumerate(make_iterable(signals_side)):
                    meta_data['signal'] = signal


                    data_plot, artist = plot_mixed_fire_uda_signal(signal, path_data=path_data, meta_data=meta_data,
                                                                    ax=ax_left, normalise_factor=normalise_factor,
                                                                   label=label, plot_kwargs=dict(alpha=alpha),
                                                                   legend=False)
                    if normalise_sigs_on_same_ax and (normalise_factor is False):
                        normalise_factor = np.max(data_plot)  # For subsequent signals on same side axis, normalise
                    # TODO: Update legend to give normalisation
                if separate_pulse_axes:
                    if (len(signals_side) > 1) and (i_ax_col == n_ax_cols-1):
                        plot_tools.legend(ax=ax_left, only_multiple_artists=False, box=True)
                else:
                    if n_pulses > 0:
                        plot_tools.legend(ax=ax_left, only_multiple_artists=False, box=True)

            # TODO: Move turning off of ax labels in grid to function
            if (sharex == 'col') and (i_ax_row != n_ax_rows-1):
                ax_left.xaxis.label.set_visible(False)
                if ax_right is not None:
                    ax_right.xaxis.label.set_visible(False)
            if (sharey == 'row') and (i_ax_col != 0):
                ax_left.yaxis.label.set_visible(False)
                # ax.set_ylabel('')
                if ax_right is not None:
                    ax_right.yaxis.label.set_visible(False)

            if x_range is not None:
                ax.set_xlim(*x_range)
    plt.tight_layout()


def identify_profile_time_highlights(profile_2d):
    raise NotImplementedError


if __name__ == '__main__':
    pass