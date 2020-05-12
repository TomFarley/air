# -*- coding: future_fstrings -*-
#!/usr/bin/env python

"""


Created: 
"""

import logging

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from fire.misc.utils import make_iterable, to_image_dataset
from fire.plotting.plot_tools import get_fig_ax
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fire.camera.image_processing import find_outlier_pixels

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

dpi = 90

cbar_label_defaults = {
                      'frame_data': r'DL [arb]',
                      'frame_temperature': r'$T$ [$^\circ$C]',
                      'R_im': r'$R$ coordinate [m]',
                      'phi_deg_im': r'$\phi$ coordinate [$^\circ$]',
                      'x_im': r'$x$ coordinate [m]',
                      'y_im': r'$y$ coordinate [m]',
                      'z_im': r'$z$ coordinate [m]',
                      's_global_im': r'$s$ coordinate [m]',
                      'sector_im': r'Sector number',
                      }

def figure_imshow(data, key='data', slice=None, ax=None,
                  add_colorbar=True, cbar_label=None, robust=True,
                  scale_factor=None,
                  cmap=None, log_cmap=False, nan_color=('red', 0.2),
                  origin='upper', aspect='equal', axes_off=False,
                  save_fn=None, show=False, **kwargs):

    fig, ax, ax_passed = get_fig_ax(ax, num=key)
    data = to_image_dataset(data, key=key)
    data_plot = data[key]
    if (data_plot.ndim > 2) and (slice is None):
        slice = {'n': np.floor(np.median(data_plot['n']))}
    if slice is not None:
        data_plot = data_plot.sel(slice)

    if cbar_label is None:
        cbar_label = cbar_label_defaults.get(key, key)

    if scale_factor is not None:
        data_plot = data_plot * scale_factor
    if log_cmap:
        norm = colors.LogNorm()
        data_plot = data_plot.where(data_plot > 0, np.nan)  # Remove zeros to avoid divide by zero
    else:
        norm = None
    kws = {}
    if add_colorbar:
        # Force xarray generated colorbar to only be hight of image axes and thinner
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        kws.update(dict(cbar_kwargs=dict(label=cbar_label, cax=cax),  # , extend='both'
                        robust=robust))
        if np.issubdtype(data_plot.dtype, np.int64) and (np.max(data_plot.values) < 20):
            # Integer/quantitative data
            bad_value = -1
            data_plot = xr.where(data_plot == bad_value, np.nan, data_plot)

            values = data_plot.values.flatten()
            values = set(list(values[~np.isnan(values)]))
            ticks = np.sort(list(values))
            kws['cbar_kwargs']['ticks'] = ticks
            kws['vmin'] = np.min(ticks)
            kws['vmax'] = np.max(ticks)
            if cmap is None:
                cmap = 'jet'
            cmap = mpl.cm.get_cmap(cmap, (ticks[-1]-ticks[0])+1)

    kws.update(kwargs)

    if cmap is None:
        cmap = 'gray'
    if isinstance(cmap, str):
        cmap = mpl.cm.get_cmap(cmap)
    if nan_color is not None:
        nan_color = make_iterable(nan_color, cast_to=iter)
        cmap.set_bad(color=next(nan_color), alpha=next(nan_color, 0.5))

    try:
        data_plot.plot.imshow(ax=ax,
                          interpolation='none', origin=origin, cmap=cmap, norm=norm, add_colorbar=add_colorbar,
                          **kws)
    except ValueError as e:
        logger.warning(f'Failed to plot {key} image data. Data has {data_plot.ndims} dims.')
        raise
    ax.title.set_fontsize(10)

    if axes_off:
        ax.set_axis_off()
    if aspect:
        ax.set_aspect(aspect)
    if not ax_passed:
        pass
        # plt.tight_layout()

    if save_fn:
        plt.savefig(save_fn, bbox_inches='tight', transparent=True, dpi=dpi)
        logger.info(f'Saved {key} figure to: {save_fn}')
    if show:
        plt.show()

def figure_frame_data(data, ax=None, n='median', key='frame_data', label_outliers=True, aspect='equal', axes_off=True,
                      save_fn=None,
                      show=False,
                      **kwargs):
    frame_data = data[[key]]
    if n == 'median':
        n = np.floor(data[key].n.median())
    slice = {'n': n} if (n is not None) else None

    figure_imshow(frame_data, key, slice=slice, ax=ax, axes_off=axes_off, aspect=aspect,
                  save_fn=save_fn, show=show, **kwargs)
    if label_outliers:
        plot_outlier_pixels(frame_data, key=key, ax=ax, **kwargs)
    # TODO: Move show save checks here

def plot_outlier_pixels(data, key='frame_data', ax=None, n=None, color='r', ms=2, **kwargs):
    if isinstance(data, (xr.DataArray, xr.Dataset)):
        frame_data = data[key]
        if 'n' in data.dims:
            if n is None:
                n = np.floor(data[key].n.median())
            frame_data = frame_data.sel(n=n)
    else:
        frame_data = data
    hot_pixels, frame_data = find_outlier_pixels(frame_data, tol=3.5)
    ax.plot(hot_pixels[1], hot_pixels[0], ls='', marker='o', color=color, ms=ms, **kwargs)

def figure_analysis_path(path_data, image_data=None, key=None, path_names='path0', slice=None, frame_border=True, image_shape=None,
                         ax=None, show=True, image_kwargs=None):
    path_names = make_iterable(path_names)
    fig, ax, ax_passed = get_fig_ax(ax, num=f'analysis_paths')

    if image_data:
        if key is not None:
            image_kwargs = {} if image_kwargs is None else image_kwargs
            figure_imshow(image_data, key=key, slice=slice, ax=ax, show=False, **image_kwargs)
        if image_shape is None:
            if key is None:
                key = 'frame_data'
            if slice is None:
                slice = {} if image_data[key].ndim == 2 else {'n': 0}
            image_shape = image_data[key].sel(slice).shape
    # Frame outline
    if frame_border and (image_shape is not None):
        plot_frame_border(ax, image_shape)

    for path in path_names:
        # Analysis path
        # TODO: Colorcode line by visibility - use data['visible_path'
        # TODO: colorcode line by path section index
        try:
            plot_analysis_path(ax, path_data[f'x_pix_{path}'], path_data[f'y_pix_{path}'],
                               in_frame=path_data[f'in_frame_{path}'])
        except KeyError as e:
            logger.warning(f'Failed to plot analysis path: {str(e)}')
    if show:
        plt.show()
    return ax

def plot_frame_border(ax, image_shape):
    if len(image_shape) != 2:
        raise ValueError(f'Expected 2D image shape, not: {image_shape}')
    ax.add_patch(mpl.patches.Rectangle((0, 0), image_shape[1], image_shape[0], color='k', fill=False))

def plot_analysis_path(ax, xpix, ypix, in_frame=None, **kwargs):
    """Plot pixel path over image"""
    kws = dict(marker='o', ms=2, ls='-', lw=1)
    kws.update(kwargs)
    ax.plot(xpix, ypix, **kws)
    if in_frame is not None:
        kws = dict(marker='x', ms=2, ls='-', lw=1, color='r')
        kws.update(kwargs)
        ax.plot(xpix, ypix, **kws)

def figure_spatial_res_max(data, ax=None, log_cmap=True, aspect='equal', axes_off=True, save_fn=None, show=False):
    key = 'spatial_res_max'
    cbar_label = 'Spatial resolution (max) [mm]'
    scale_factor = 1e3
    cmap = 'gray_r'
    figure_imshow(data, key, cbar_label=cbar_label, ax=ax, cmap=cmap, log_cmap=log_cmap, axes_off=axes_off, aspect=aspect,
                  scale_factor=scale_factor, save_fn=save_fn, show=show)

def figure_spatial_res_x(data, ax=None, log_cmap=True, aspect='equal', axes_off=True, save_fn=None, show=False):
    key = 'spatial_res_x'
    cbar_label = 'Spatial resolution (x) [mm]'
    scale_factor = 1e3
    cmap = 'gray_r'
    figure_imshow(data, key, cbar_label=cbar_label, ax=ax, cmap=cmap, log_cmap=log_cmap, axes_off=axes_off, aspect=aspect,
                  scale_factor=scale_factor, save_fn=save_fn, show=show)

def figure_spatial_res_y(data, ax=None, log_cmap=True, aspect='equal', axes_off=True, save_fn=None, show=False):
    key = 'spatial_res_y'
    cbar_label = 'Spatial resolution (y) [mm]'
    scale_factor = 1e3
    cmap = 'gray_r'
    figure_imshow(data, key, cbar_label=cbar_label, ax=ax, cmap=cmap, log_cmap=log_cmap, axes_off=axes_off,
                  aspect=aspect, scale_factor=scale_factor, save_fn=save_fn, show=show)

def plot_image_data_hist(data, key=None, ax=None, bins=200, xlabel=None, log_x=False, log_y=True, save_fn=None,
                         show=False, **kwargs):
    fig, ax, ax_passed = get_fig_ax(ax, num=key)
    if key is not None:
        data_plot = data[key].values
    else:
        data_plot = np.array(data)
    ax.hist(data_plot.flatten(), bins=bins, **kwargs)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    if not ax_passed:
        plt.tight_layout()
        ax.set_aspect('equal')
    if save_fn:
        plt.savefig(save_fn, bbox_inches='tight', transparent=True, dpi=dpi)
        logger.info(f'Saved y image histogram figure of {key} to: {save_fn}')
    if show:
        plt.show()

def figure_poloidal_cross_section(image_data=None, path_data=None, path_names='path0', ax=None,
                                  closest_points=True, tile_edges=True, legend=True, axes_off=False,
                                  pulse=50000, no_cal=False, show=True):
    import warnings
    from fire.geometry.s_coordinate import get_nearest_boundary_coordinates
    if fire.active_machine_plugin is not None:
        (machine_plugins, machine_plugins_info) = fire.active_machine_plugin
        plot_vessel_outline = machine_plugins['plot_vessel_outline']
        get_wall_rz_coords = machine_plugins['get_wall_rz_coords']
    else:
        from fire.plugins.machine_plugins.mast_u import plot_vessel_outline_mastu as plot_vessel_outline
        from fire.plugins.machine_plugins.mast_u import plot_tile_edges_mastu
        from fire.plugins.machine_plugins.mast_u import get_wall_rz_coords

    path_names = make_iterable(path_names)
    fig, ax, ax_passed = get_fig_ax(ax, num='pol cross section')

    if image_data is not None:
        r_im, z_im = np.array(image_data['R_im']).flatten() ,np.array(image_data['z_im']).flatten()
    if path_data is not None:
        # Get full z extent of path to inform whether to plot top and/or bottom of machine
        z_path_all = []
        for path in path_names:
            z_path_all.append(np.array(path_data[f'z_{path}']))
        z_path_all = np.concatenate(z_path_all)

    if image_data or path_data:
        z = z_im if image_data else z_path_all
        with warnings.catch_warnings():   # ignore nan comparison
            warnings.simplefilter("ignore")
            top = True if np.any(z > 0) else False
            bottom = True if np.any(z < 0) else False
    else:
        top, bottom = True, True


    plot_vessel_outline(ax=ax, top=top, bottom=bottom, shot=pulse, no_cal=no_cal, show=False, label='Wall',
                              axes_off=axes_off)
    if tile_edges is True:
        plot_tile_edges_mastu(ax=ax, top=top, bottom=bottom, markersize=4, color='k', label='Tile edges', show=False)

    # TODO: Plot rays from camera pupil showing field of view

    if image_data is not None:
        # Plot the poloidal cross section of all the surfaces visible in the images
        ax.plot(r_im, z_im, ls='', marker='o', markersize=2, color='orange', label='Image pixels')
        if closest_points:
            r_wall, z_wall = get_wall_rz_coords(no_cal=no_cal, shot=pulse, ds=1e-3)
            closest_coords, closest_dist, closest_index = get_nearest_boundary_coordinates(r_im, z_im, r_wall, z_wall)
            r_close, z_close = closest_coords[:, 0], closest_coords[:, 1]
            ax.plot(r_close, z_close, ls='', marker='o', markersize=1, color='purple', alpha=0.6,
                    label='Wall closest to image pixels')
            print(f'Dists for all image coords: min: {np.nanmin(closest_dist):0.3g}, '
                                              f'mean: {np.nanmean(closest_dist):0.3g}, '
                                              f'max: {np.nanmax(closest_dist):0.3g}')

    if path_data is not None:
        # Plot the poloidal cross section of the surfaces along the analysis path(s)
        for path in path_names:
            r_path, z_path = np.array(path_data[f'R_{path}']), np.array(path_data[f'z_{path}'])
            ax.plot(r_path, z_path, ls='', marker='o', markersize=2, color='green', label='Analysis path pixels', alpha=0.7)
            if closest_points:
                r_wall, z_wall = get_wall_rz_coords(no_cal=no_cal, shot=pulse, ds=1e-4)
                closest_coords, closest_dist, closest_index = get_nearest_boundary_coordinates(r_path, z_path, r_wall, z_wall)
                r_close, z_close = closest_coords[:, 0], closest_coords[:, 1]
                ax.plot(r_close, z_close, ls='', marker='o', markersize=2, color='red', alpha=0.5,
                        label=f'Wall closest to analysis {path} pixels')
                print(f'Dists for {path} coords: min: {np.nanmin(closest_dist):0.3g}, '
                                                f'mean: {np.nanmean(closest_dist):0.3g}, '
                                                f'max: {np.nanmax(closest_dist):0.3g}')
            pass
    if legend:
        kws = {'fontsize': 7, 'framealpha': 0.7, 'facecolor': 'white', 'fancybox': True}
        leg = ax.legend(**kws)
        leg.set_draggable(True)

    plt.tight_layout()

    if show:
        plt.show()

if __name__ == '__main__':
    pass