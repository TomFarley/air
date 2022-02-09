# -*- coding: future_fstrings -*-
#!/usr/bin/env python

"""


Created: 
"""

import logging
from pathlib import Path
from copy import copy, deepcopy
from functools import partial

import numpy as np

from matplotlib import cm
import matplotlib.pyplot as plt
from cycler import cycler

from fire.plotting import image_figures, spatial_figures, path_figures, plot_tools, temporal_figures
from fire.plotting.image_figures import (figure_xarray_imshow, figure_frame_data, plot_outlier_pixels, figure_analysis_path,
                                         figure_spatial_res, figure_spatial_res_x, figure_spatial_res_y,
                                         plot_image_data_hist, plot_analysis_path, animate_image_data)
from fire.plotting.spatial_figures import figure_poloidal_cross_section, figure_top_down
from fire.plotting.path_figures import figure_path_1d, figure_path_2d
from fire.plotting.plot_tools import annotate_axis, repeat_color
from fire.camera_tools.image_processing import find_outlier_pixels
from fire.plugins import plugins, plugins_machine
from fire.misc import utils, data_structures
from fire.misc.utils import make_iterable
from fire.interfaces import uda_utils
from fire.misc.data_structures import select_variable_from_dataset

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
    figure_xarray_imshow(data, 'nuc_frames', slice_={'i_digitiser': 0}, ax=ax, axes_off=True, aspect=aspect,
                            show=False)

    ax = axes[4]
    nuc_frame_0 = data['nuc_frames'].sel(i_digitiser=0)
    plot_image_data_hist(nuc_frame_0, key=None, xlabel='NUC frame 0 DL', ax=ax, show=False)

    ax = axes[5]
    image_figures.plot_image_data_temporal_stats(data, key=key, ax=ax,
                                                 x_key='n', stats=('max', 'mean', 'median', 'min'))
    plt.tight_layout()
    plt.show()

def debug_calcam_calib_image(calcam_calib, frame_data=None, frame_ref=None, n_frame_ref=None, wire_frame=None):
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
    fig, axes = plt.subplots(*fig_shape, num='calcam_calib_image', figsize=(14, 8))  # , sharex='col', sharey='col')
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

    if wire_frame is not None:
        ax = axes[i_ax]

        if wire_frame.ndim > 2:
            wire_frame[:, :, 1:2] = 0  # Zero out green and blue

        ax.imshow(frame_display, interpolation='none', cmap='gray', origin='upper')
        ax.imshow(wire_frame, interpolation='none', origin='upper', alpha=0.7, cmap='Reds')  # Cmap ignored for RCB(A)
        ax.set_title(f'Wire frame overlaid movie image: {Path(calcam_calib.filename).name}',
                     fontdict={'fontsize': title_size})
        i_ax += 1


    plt.tight_layout()
    plt.show()

def debug_detector_window(detector_window, frame_data=None, key='frame_data_nuc', calcam_calib=None,
                          image_full_frame=None, subview_mask_full_frame=None, aspect='equal',
                          image_coords='Display', image_full_frame_label=None, meta_data=None):
    if meta_data is None:
        meta_data = frame_data.attrs.get('meta_data', {})

    if image_full_frame is None:
        # Get full frame calibration image without detector window applied
        calcam_calib.set_detector_window(None)
        image_full_frame = calcam_calib.get_image(coords=image_coords)
        subview_mask_full_frame = calcam_calib.get_subview_mask(coords=image_coords)
        calcam_calib.set_detector_window(detector_window)  # Re-apply detector_window
        image_full_frame_label = f'Calcam calibration image ({calcam_calib.name})'

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
        plot_tools.annotate_providence(ax, meta_data=meta_data)

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
                        coord_keys=('x_im', 'y_im',
                                    'R_im', 'phi_deg_im', 'z_im',
                                    'ray_lengths_im', 'sector_im', 'wire_frame'),
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

    for ax, key in zip(axs, coord_keys):
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
    figure_spatial_res(data, ax=ax, show=False, save_fn=None, aspect=aspect)

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
                    if key in path_data:
                        spatial_figures.figure_spatial_profile_1d(path_data, key, path_name=path, ax=ax,
                                                              plot_kwargs=plot_kwargs, legend=(nlines > 1), show=False)
                    else:
                        logger.warning(f'Cannot plot spatial profile for missing data: "{key}"')

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

        spatial_figures.figure_poloidal_cross_section(image_data=im_data, path_data=path_data, path_names=path_names,
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
    plot_tools.save_fig(path_fn_save, mkdir_depth=3)
    return fig


def debug_temperature_image(data):
    figure_xarray_imshow(data, key='temperature_im', show=True)

def debug_plot_spatial_2d_unwrapped(image_data, key='frame_data', spatial_dims=('R_im', 'phi_im'), ax=None, show=True):

    fig, ax = plot_tools.get_fig_ax(ax=None)

    data = image_data[key]
    x = data[spatial_dims[0]]
    y = data[spatial_dims[1]]

    ax.plot(x, y, data)
    plot_tools.show_if(show=show)
    raise NotImplementedError


def debug_plot_profile_2d(data_paths, param='temperature', path_names='path0', coord_path='R_{path_name}',
                          robust=True, extend=None,
                          annotate=True, mark_peak=True, label_tiles=True, meta=None, ax=None, t_range=(0.0, 0.6),
                          x_range=None, t_wins=None, machine_plugins=None, add_colorbar=True, colorbar_kwargs=None,
                          set_data_coord_lims_with_ranges=True, robust_percentiles=(2, 98), num=None,
                          show=True, save_path_fn=None, image_formats=('png'), verbose=True):
    # TODO: Move general code to plot_tools.py func
    path_names = make_iterable(path_names)
    ax_in = ax
    for i_path, path_name in enumerate(path_names):
        num = f'{param}_profile_2d {path_name}' if (num is None) else num
        fig, ax, ax_passed = plot_tools.get_fig_ax(ax=ax_in, ax_grid_dims=(len(path_names), 1), num=num)
        ax_i = make_iterable(ax)[i_path]

        data = data_paths[f'{param}_{path_name}']

        coord_path = coord_path.format(path_name=path_name)
        # coord_path = f's_{path_name}'
        data = data_structures.swap_xarray_dim(data, new_active_dims=['t', coord_path])
        data_masked = copy(data)

        # TODO: Generalise setting coord ranges to other 2d data
        if t_range is not None:
            t_range = np.array(t_range)
            t_range[1] = np.min([t_range[1], data['t'].values.max()]) if (t_range[1] is not None) else (data[
                                                                                                't'].values.max())
            if set_data_coord_lims_with_ranges:
                mask = (t_range[0] <= data['t']) & (data['t'] <= t_range[1])
                data_masked = data.where(mask)
                # data = data.sel({'t': slice(*t_range)})
            ax_i.set_ylim(*t_range)
        if x_range is not None:
            x_range = np.array(x_range)
            x_range[0] = np.max([x_range[0], data[coord_path].values.min()]) if (x_range[0] is not None) else (data[
                                                                                            coord_path].values.min())
            x_range[1] = np.min([x_range[1], data[coord_path].values.max()])
            if set_data_coord_lims_with_ranges:  # set data ranges to get full range from colorbar
                # TODO: Make general purpose function for slicing coords since pandas bug or update packages? https://github.com/pydata/xarray/issues/4370
                # mask = (r_range[0] <= data[coord_path]) & (data[coord_path] <= r_range[1])
                # data_masked = data[mask]
                data_masked = data.sel({coord_path: slice(*x_range)})
            ax_i.set_xlim(*x_range)

        # Configure colorbar axis
        if robust:
            mask_nan = np.isnan(data_masked)
            vmin, vmax = (np.percentile(data_masked.values[~mask_nan], robust_percentiles[0]),
                          np.percentile(data_masked.values[~mask_nan], robust_percentiles[1]))
        else:
            vmin, vmax = np.min(data_masked), np.max(data_masked)
        colorbar_kwargs = colorbar_kwargs if (colorbar_kwargs is not None) else {}
        cmap = cm.get_cmap('coolwarm')
        colorbar_kws = dict(robust=robust, extend=extend, cmap=cmap, vmin=vmin, vmax=vmax)
        colorbar_kws.update(colorbar_kwargs)
        kws = plot_tools.setup_xarray_colorbar_ax(ax_i, data_plot=data, add_colorbar=add_colorbar, **colorbar_kws)

        # Plot data
        try:
            # Plot all data so you can pan outside axis limit ranges but set colorbar ranges using data_masked
            # Remove any nans from path coord as will make 2d plot fail
            mask_coord_nan = np.isnan(data[coord_path])
            if np.any(mask_coord_nan):
                data = data.sel({coord_path: ~mask_coord_nan})
            try:
                r_dim = data.dims[1]
                data = data.sortby(r_dim, ascending=True)
                artist = data.plot.imshow(center=False, ax=ax_i, **kws)  # pcolormesh
            except ValueError as e:
                # contourf can handle irregular x axis
                artist = data.plot.contourf(levels=200, center=False, ax=ax_i, **kws)
        except (KeyError, ValueError) as e:
            # Value error due to: R data not monotonic or nans in coord values - switch back to index or s_path
            data = data_structures.swap_xarray_dim(data, f'i_{path_name}')

            artist = data.plot(center=False, ax=ax_i, **kws)

        # Mark peak
        param_peak = f'{param}_amplitude_peak_global_{path_name}'
        param_r_peak = f'{param}_R_peak_{path_name}'
        if mark_peak and (param_r_peak in data_paths):
            # TODO: filter instead with moving window stdev of r pos?
            data_peak = data_paths[param_peak].values
            data_r_peak = data_paths[param_r_peak]
            mask_low = data_peak < (np.nanmin(data_peak) + 0.01 * (np.nanmax(data_peak)-np.nanmin(data_peak)))
            data_r_peak = data_r_peak.where(~mask_low, np.nan)
            ax_i.plot(data_r_peak.values, data_r_peak[data.dims[0]], ls='', marker='x', ms=2, color='g',  # 'g'
                      lw=2.5, alpha=0.4, label='peak')  # ms=1.5

        if annotate and (meta is not None):
            # TODO: Make figure pulse label func
            # plot_tools.annotate_providence(ax_i, loc='top_right', meta=meta, box=False)
            plot_tools.annotate_providence(ax_i, loc='top left', meta_data=meta, box=False)

            # label = f'{meta["machine"]} {meta["camera"]} {meta["pulse"]}'.replace('_u', '-U').upper()
            # annotate_axis(ax_i, label, x=0.99, y=0.99, fontsize=12, box=False, horizontalalignment='right',
            #                                                                             verticalalignment='top')

        if t_range is not None:
            ax_i.set_ylim(*t_range)
        if x_range is not None:
            ax_i.set_xlim(*x_range)
        else:
            ax_i.set_xlim(data[coord_path].values.min(), data[coord_path].values.max())

        if label_tiles:
            plugins.get_and_call_plugin_func('label_tiles', plugins_dict=machine_plugins, plugin_type='machine',
                                             plugin_defaults=dict(machine='mast_u'),
                                             kwargs_plugin=dict(ax=ax, coords=data.coords, path_no=0))

        if t_wins is not None:  # Add horizontal lines labeling times
            plot_tools.label_axis_windows(windows=t_wins, labels=t_wins, ax=ax_i, axis='y', line_kwargs=None)

        if extend == 'min':
            pass
            # cmap.set_over(data.max())

        # TODO: Switch to using same routine as for uda_utils.plot_uda_dataarray
        if verbose:
            logger.info(f'{param}({path_name}): min={data.min().values:0.4g}, mean={data.mean().values:0.4g}, '
                        f'99%={np.percentile(data.values,99):0.4g}, max={data.max().values:0.4g}')

        plot_tools.save_fig(save_path_fn, image_formats=image_formats, fig=fig, mkdir_depth=3, meta_dict=meta)
        plot_tools.show_if(show=show, tight_layout=True)

    return ax, data, artist

def debug_plot_spatial_profile_1d(data_paths, params='temperature', path_name='path0', ax=None,
                                  label_tiles=True, meta_data=None, machine_plutings=None, show=True, plot_kwargs=()):
    params = make_iterable(params)
    plot_kwargs = dict(plot_kwargs)
    kws = {'alpha': 0.8}
    kws.update(plot_kwargs)

    fig, ax, ax_passed = plot_tools.get_fig_ax(ax, num=f'spatial profiles {params} {path_name}')

    for i_param, param in enumerate(params):
        ax_active, ax_other = plot_tools.add_second_y_scale(ax_left=ax, apply=(i_param == 1))

        data, param = data_structures.select_variable_from_dataset(data_paths, variable_name=param,
                                                                   path_name=path_name, i_path=0)

        data = data_structures.swap_xarray_dim(data, ('t', f'R_{path_name}'), raise_on_fail=False)

        # t_slices = identify_profile_time_highlights() if t is None else t

        data.plot(ax=ax_active, **kws)

    if meta_data is not None:
        plot_tools.annotate_providence(ax, meta_data=meta_data)

    if label_tiles:
        plugins.get_and_call_plugin_func('label_tiles', plugin_type='machine',
                        kwargs_plugin=dict(ax=ax, coords=data.coords, path_no=0))

    plot_tools.show_if(show=show, tight_layout=True)

    return fig, ax

def debug_plot_temporal_profile_1d(data_paths, params=('heat_flux_R_peak', 'heat_flux_amplitude_global_peak'), path_name='path0',
                                   x_var='t', heat_flux_thresh=-0.0, meta_data=None, machine_plugins=None, ax=None,
                                   show=True):
    # TODO: Move general code to plot_tools.py func
    # TODO: Move multi-signal plotting on same axis to general function for multiaxes etc
    colors = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red')
    path = path_name if isinstance(path_name, str) else path_name[0]
    params = make_iterable(params)
    n_params = len(params)

    fig, ax1, ax_passed = plot_tools.get_fig_ax(num=f'{params} temporal profile {path}', ax=ax)
    ax = ax1
    if n_params > 1:
        ax.tick_params(axis='y', labelcolor=colors[0])

    if heat_flux_thresh not in (None, False):
        peak_heat_flux, key = select_variable_from_dataset(data_paths, 'heat_flux_amplitude_global_peak',
                                                      path_name=path_name)
        mask_pos_heat_flux = peak_heat_flux > heat_flux_thresh
        peak_heat_flux_pos = peak_heat_flux[mask_pos_heat_flux]
    else:
        mask_pos_heat_flux = np.ones_like(data_paths['t'].data, dtype=bool)

    for i, (param, color) in enumerate(zip(params, colors)):
        if i == 1:
            ax = ax1.twinx()
            ax.tick_params(axis='y', labelcolor=color)

        data, key = select_variable_from_dataset(data_paths, param, path_name=path_name)

        if x_var not in data.dims:
            data = data.swap_dims({data.dims[0]: x_var})
        y_var = data.name

        data_pos_q = deepcopy(data)
        data_pos_q[~mask_pos_heat_flux] = np.nan

        try:
            if heat_flux_thresh is not None:
                data.plot(label=f'{param} (all)', ax=ax, ls=':', alpha=0.8, color=color)
                temporal_figures.plot_temporal_profile(data_paths, key, path_name=path, mask=mask_pos_heat_flux,
                                        label=f'{param} (pos q)', ax=ax, ls='-', alpha=0.6, color=color, show=False)
            else:
                label = plot_tools.format_label(param)
                temporal_figures.plot_temporal_profile(data_paths, key, path_name=path, mask=mask_pos_heat_flux,
                                        label=label, ax=ax, ls='-', alpha=0.6, color=color, show=False)
        except ValueError as e:
            raise NotImplementedError
            # R data not monotonic - switch back to index or s_path
            # data = data.sortby('')
            # data = data.swap_dims({f'R_{path_name}': f's_path_{path_name}'})
            data = data.swap_dims({f'R_{path}': f'i_{path}'})
            data.plot(robust=True, center=False, cmap='coolwarm')
        else:
            plot_tools.legend(ax=ax)
        if ('r_peak' in y_var) and (machine_plugins is not None):
            label_tiles = machine_plugins.get('label_tiles')
            if label_tiles is not None:
                label_tiles(ax, coords=data, coords_axes=(data.dims[0],))

    if heat_flux_thresh not in (None, False):
        ax.axhline(heat_flux_thresh, ls='--', color='k')
        ax.set_xlim([peak_heat_flux_pos['t'].min(), peak_heat_flux_pos['t'].max()])

    if meta_data is not None:
        plot_tools.annotate_providence(ax, meta_data=meta_data)

    plot_tools.show_if(show=show, tight_layout=True)

    return fig, ax

def plot_energy_to_target(data_paths, params=('heat_flux_R_peak', 'heat_flux_amplitude_global_peak'), path_name='path0',
                          meta_data=None, machine_plugins=None):
    path = path_name if isinstance(path_name, str) else path_name[0]
    params = make_iterable(params)
    n_params = len(params)

    ax_grid_dims = plot_tools.get_ax_grid_dims(n_ax=n_params, n_max_ax_per_row=3)

    fig, axes, ax_passed = plot_tools.get_fig_ax(num=f'{params} temporal profile {path}', ax_grid_dims=ax_grid_dims,
                                                axes_flatten=True, sharex='col')
    for i_ax, param in enumerate(params):
        ax = axes[i_ax]
        if i_ax == 0:
            plot_tools.annotate_providence(ax, meta_data=meta_data)

        if param in ('power_total_vs_t', 'cumulative_energy_vs_t'):  # TODO: Check dim instead of hard list
            debug_plot_temporal_profile_1d(data_paths, params=param, ax=ax, heat_flux_thresh=None,
                                           meta_data=None, show=False)
        elif param in ('energy_total_vs_R', 'cumulative_energy_vs_R'):
            debug_plot_spatial_profile_1d(data_paths, params=param, ax=ax, path_name=path_name, meta_data=None,
                                                                                                   show=False)
        else:
            raise NotImplementedError

    plot_tools.annotate_providence(ax, meta_data=meta_data)

    plot_tools.show_if(show=True, tight_layout=True)

    return fig, axes

def debug_plot_timings(data_profiles, pulse, params=('heat_flux_amplitude_global_peak_{path}',
                                                     'temperature_amplitude_peak_global_{path}',),
                       path_name='path0', comparison_signals=(('xim/da/hm10/t', 'xim/da/hm10/r'),
                                                               'xpx/clock/lwir-1'), separate_axes=True,
                       meta_data=None):
    from fire.interfaces import uda_utils
    from fire.physics.physics_parameters import find_peaks_info
    # uda_module, client = uda_utils.get_uda_client(use_mast_client=True, try_alternative=True)

    n_peaks_label = 3
    figsize = (10, 10)
    ylabel_size = 12

    if not separate_axes:
        n_axes = 1
    else:
        n_axes = len(params) + len(comparison_signals)

    fig, axes, ax_passed = plot_tools.get_fig_ax(num='timings check', figsize=figsize, ax_grid_dims=(n_axes, 1),
                                                                                               sharex=True)#'col')
    axes = make_iterable(axes, cast_to=np.ndarray).flatten()
    i_ax = 0

    plot_tools.annotate_providence(axes[i_ax], loc='top right', meta_data=meta_data, annotate=(meta_data is not None))

    t_peaks = []
    frame_times = data_profiles[params[0].format(path=path_name)]['t']

    for signals in make_iterable(comparison_signals):
        ax = axes[i_ax]
        signals = make_iterable(signals)
        n_sigs = len(signals)
        for i_sig, sig in enumerate(signals):
            data = uda_utils.read_uda_signal_to_dataarray(sig, pulse=pulse, raise_exceptions=False)
            if isinstance(data, Exception):
                logger.warning(f'Failed to read uda signal {sig} for {pulse}: {data}')
                continue
            peaks_info = find_peaks_info(data, peak_kwargs=(('width', 5),))
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
            ax.plot(frame_times, np.zeros_like(frame_times), ls='', marker='x', markersize=4, color='k', alpha=0.5)
            i_ax += 1

    for param in make_iterable(params):
        key = param.format(path=path_name)
        data = data_profiles[key]
        peaks_info = find_peaks_info(data)
        t_peaks.append(peaks_info['x_peaks'].values[:n_peaks_label])  # Record time of 3 largest peaks

        ax = axes[i_ax]
        # param.format(path=path_name)
        temporal_figures.plot_temporal_profile(data_profiles, param=param, path_name=path_name, ax=ax,
                                                             show=False, label=True)
        plot_tools.legend(ax)

        if separate_axes:
            ax.yaxis.label.set_size(ylabel_size)
            ax.get_xaxis().set_visible(False)
            ax.plot(frame_times, np.zeros_like(frame_times), ls='', marker='x', markersize=4, color='k', alpha=0.5)
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
        label = utils.format_str_partial(label, meta_data, allow_partial=True)

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

def plot_mixed_fire_uda_signals(signals, pulses=None, path_data=None, meta_data=None, axes=None,
                                sharex=None, sharey='row', x_range=None, separate_pulse_axes=True,
                                normalise_sigs_on_same_ax=True, recompute_pickle=False,
                                slice_labels=True, **kwargs):
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
        # label = '{signal}'  # format string
        annotate_format = '{machine} {diag_tag_raw} {pulse} {path_label}'
    else:
        n_ax_cols = 1
        # label = '{pulse}'  # format string
        annotate_format = '{machine} {diag_tag_raw}'

    n_ax_grid = (n_ax_rows, n_ax_cols)
    n_ax = n_ax_rows * n_ax_cols
    figsize = (14, 8) if (n_ax > 1) else (8, 5)

    fig, axes, ax_passed = plot_tools.get_fig_ax(ax=axes, ax_grid_dims=n_ax_grid, sharex=sharex, sharey=sharey,
                                                 figsize=figsize)
    axes = np.array(make_iterable(axes))
    if n_ax_cols == 1:
        axes = axes[..., np.newaxis]

    slice_ = kwargs.get('slice_')
    slice_str = '' if ((slice_ is None) or (not slice_labels)) else (' '+' '.join(['{'+f'{key}'+'}' for key in
                                                                                 slice_.keys()]))

    # TODO: enable plotting same signal at different times for same pulse
    for i_pulse, pulse in enumerate(pulses):
        if isinstance(pulse, (tuple, list)):
            kwargs.update(pulse[1])
            pulse = pulse[0]
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
                data = read_data_for_pulses_pickle(camera, pulse, machine, recompute=recompute_pickle)
                path_data = data[pulse][0]['path_data']
            except (FileNotFoundError, KeyError) as e:
                pass

        # TODO: Make separate functions for setting up axes, labels, formatting etc based on input

        for i_ax_row, signals_ax in enumerate(signals):
            # TODO: Allow slice_ and reduce to be iterables with length of number of axes or dict keyed by
            # signal/shot and pull them out here?
            # TODO: OR allow signal and/or pulse list items (in list) to be dict with specific settings
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
                        label = '{signal}'+slice_str
                    else:
                        if (n_pulses > 1) and (n_signals_side > 1):
                            label = '{pulse} {signal}'+slice_str
                        else:
                            label = '{pulse}'+slice_str

                    alpha = 1 if (len(signals_side) == 1) and (n_pulses == 1 or separate_pulse_axes) else 0.7
                    normalise_factor = False

                    for i, (signal, ls) in enumerate(zip(signals_side, line_styles)):
                        meta_data['signal'] = signal

                        kws = dict(meta_data=meta_data, normalise_factor=normalise_factor, label=label,
                                   plot_kwargs=dict(color=color, alpha=alpha, **ls), legend=False)
                        kws.update(kwargs)

                        # Plot signal
                        data_plot, artist = plot_mixed_fire_uda_signal(signal, path_data=path_data, ax=ax, **kws)

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
                    label = '{signal}'+slice_str
                else:
                    if (n_pulses > 1) and (n_signals_side > 1):
                        label = '{pulse} {signal}'+slice_str
                    else:
                        label = '{pulse}'+slice_str
                alpha = 1 if (len(signals_side) == 1) and (n_pulses == 1 or separate_pulse_axes) else 0.7
                normalise_factor = False
                for i_left, signal in enumerate(make_iterable(signals_side)):
                    meta_data['signal'] = signal

                    kws = dict(meta_data=meta_data, normalise_factor=normalise_factor, label=label,
                                                        plot_kwargs=dict(alpha=alpha), legend=False)
                    kws.update(kwargs)

                    # Plot signal
                    data_plot, artist = plot_mixed_fire_uda_signal(signal, path_data=path_data, ax=ax_left, **kws)

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

            ax.title.set_visible(False)
    plt.tight_layout()

def plot_bad_pixels(mask_bad_pixels, frame_data=None, frame_data_corrected=None, alpha=0.8, cmap='Reds_r'):

    n_ax = 1 + (frame_data is not None) + (frame_data_corrected is not None)
    fig, axes, ax_passed = plot_tools.get_fig_ax(ax_grid_dims=(2, n_ax), num='bad pixels', axes_flatten=True,
                                                 sharex=True, sharey=True)

    ax = axes[0]
    ax.imshow(mask_bad_pixels)
    ax.set_title(f'Bad pixel mask ({np.sum(mask_bad_pixels)})')

    if frame_data is not None:
        ax = axes[1]
        image_figures.figure_xarray_imshow(frame_data, ax=ax)
        # overplot_image(mask_bad_pixels, ax, alpha=alpha, cmap=cmap, nan_threshold=0)
        ax.set_title(f'Raw image')

    if frame_data_corrected is not None:
        ax = axes[2]
        image_figures.figure_xarray_imshow(frame_data_corrected, ax=ax)
        # overplot_image(mask_bad_pixels, ax, alpha=alpha, cmap=cmap, nan_threshold=0)
        ax.set_title(f'BPR corrected image')

    ax = axes[n_ax]
    ax.imshow(mask_bad_pixels)
    overplot_image(mask_bad_pixels, ax, alpha=alpha, cmap=cmap, nan_threshold=0)

    if frame_data is not None:
        ax = axes[n_ax+1]
        image_figures.figure_xarray_imshow(frame_data, ax=ax)
        overplot_image(mask_bad_pixels, ax, alpha=alpha, cmap=cmap, nan_threshold=0)

    if frame_data_corrected is not None:
        ax = axes[n_ax+2]
        image_figures.figure_xarray_imshow(frame_data_corrected, ax=ax)
        overplot_image(mask_bad_pixels, ax, alpha=alpha, cmap=cmap, nan_threshold=0)

    plot_tools.show_if(tight_layout=True)

def overplot_image(image, ax, alpha=0.7, cmap=None, nan_threshold=None):
    if nan_threshold is not None:
        # Set values to nan so transparent
        image = np.where(image, image > nan_threshold, np.nan)
    ax.imshow(image, interpolation='none', origin='upper', alpha=alpha, cmap=cmap)

def identify_profile_time_highlights(profile_2d):
    raise NotImplementedError


if __name__ == '__main__':
    pass