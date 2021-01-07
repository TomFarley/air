# -*- coding: future_fstrings -*-
#!/usr/bin/env python

"""Functions for plotting data in the form of images i.e. plotting data for each pixel in the camera view


Created: 
"""

import logging
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
from fire.misc.utils import make_iterable, to_image_dataset
from fire.plotting.plot_tools import get_fig_ax, legend
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from fire.camera.image_processing import find_outlier_pixels

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

dpi = 90

# TODO: Add label information to xarray data attrs?
cbar_label_defaults = {
                      'frame_data': r'DL [arb]',
                      'frame_data_nuc': r'DL [arb]',
                      'frame_temperature': r'$T$ [$^\circ$C]',
                      'R_im': r'$R$ coordinate [m]',
                      'phi_deg_im': r'$\phi$ coordinate [$^\circ$]',
                      'x_im': r'$x$ coordinate [m]',
                      'y_im': r'$y$ coordinate [m]',
                      'z_im': r'$z$ coordinate [m]',
                      's_global_im': r'$s$ coordinate [m]',
                      'sector_im': r'Sector number',
                      'ray_lengths_im': r'Distance from camera [m]',
                      }

def figure_xarray_imshow(data, key='data', slice=None, ax=None,
                         add_colorbar=True, cbar_label=None, robust=True,
                         scale_factor=None, clip_range=(None, None),
                         cmap=None, log_cmap=False, nan_color=('red', 0.2),
                         origin='upper', aspect='equal', axes_off=False,
                         save_fn=None, show=False, **kwargs):
    # TODO: Move non-xarray specific functionality to function that can be called with numpy arrays
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

    if (clip_range is not None) and (isinstance(clip_range, (tuple, list, np.ndarray))):
        # Set rescaled data outside of clip range to nan
        if clip_range[0] is not None:
            data_plot = data_plot.where(data_plot >= clip_range[0], np.nan)
        if clip_range[1] is not None:
            data_plot = data_plot.where(data_plot <= clip_range[1], np.nan)

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
                      save_fn=None, show=False, **kwargs):
    data = xr.Dataset(data_vars=dict(key=data)) if isinstance(data, xr.DataArray) else data
    frame_data = data[key]
    if n == 'median':
        n = np.floor(frame_data.n.median())
    elif n == 'bright':
        n = frame_data.mean(dim=('x_pix', 'y_pix')).argmax(dim='n', skipna=True)
    slice = {'n': n} if (n is not None) else None

    figure_xarray_imshow(data, key, slice=slice, ax=ax, axes_off=axes_off, aspect=aspect,
                         save_fn=save_fn, show=show, **kwargs)
    if label_outliers:
        plot_outlier_pixels(data, key=key, ax=ax, **kwargs)
    # TODO: Move show save checks here

def plot_outlier_pixels(data, key='frame_data', ax=None, n=None, color='r', ms=2, **kwargs):
    if isinstance(data, (xr.DataArray, xr.Dataset)):
        frame_data = data[key] if isinstance(data, xr.Dataset) else data
        if 'n' in frame_data.dims:
            if n is None:
                n = np.floor(frame_data.n.median())
            frame_data = frame_data.sel(n=n)
    else:
        frame_data = data

    hot_pixels, frame_data = find_outlier_pixels(frame_data, tol=3.5)
    ax.plot(hot_pixels[1], hot_pixels[0], ls='', marker='o', color=color, ms=ms, alpha=0.35, **kwargs)

def plot_rzphi_points(calcam_calib, points_rzphi, ax=None, angle_units='degrees', image_coords='Display',
                      plot_kwargs=None):
    from fire.interfaces.calcam_calibs import toroidal_to_pixel_coordinates

    logger.debug('Plotting r,z,phi points on image')

    fig, ax, ax_passed = get_fig_ax(ax, num=f'Projected points')

    points_pix, info = toroidal_to_pixel_coordinates(calcam_calib, points_rzphi,
                                                     angle_units=angle_units, image_coords=image_coords,
                                                     raise_on_out_of_frame=False)

    plot_kwargs = {'ls': '', 'marker': 'x', 'color': 'r'} if plot_kwargs is None else plot_kwargs
    ax.plot(points_pix[:, 0], points_pix[:, 1], **plot_kwargs)

def figure_analysis_path(path_data, image_data=None, key=None, path_names='path0', slice=None, frame_border=True,
                         image_shape=None, color_path='green',
                         ax=None, show=True, image_kwargs=None):
    path_names = make_iterable(path_names)
    fig, ax, ax_passed = get_fig_ax(ax, num=f'analysis_paths')

    if image_data:
        if key is not None:
            image_kwargs = {} if image_kwargs is None else image_kwargs
            figure_xarray_imshow(image_data, key=key, slice=slice, ax=ax, show=False, **image_kwargs)
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
                                   xpix_out_of_frame=path_data[f'x_pix_out_of_frame_{path}'],
                                   ypix_out_of_frame=path_data[f'y_pix_out_of_frame_{path}'], color=color_path)
        except KeyError as e:
            logger.warning(f'Failed to plot analysis path: {str(e)}')
    if show:
        plt.show()
    return ax

def plot_frame_border(ax, image_shape):
    if len(image_shape) != 2:
        raise ValueError(f'Expected 2D image shape, not: {image_shape}')
    ax.add_patch(mpl.patches.Rectangle((0, 0), image_shape[1], image_shape[0], color='k', fill=False))

def plot_detector_window(ax, detector_window):
    if len(detector_window) != 4:
        raise ValueError(f'Expected 4 element array (left, right, width, height), not: {detector_window}')
    detector_window = np.array(detector_window)
    # 0.5 pixel offset becuase inshow plots pixels centred on pixel coordinate
    top_left_corner = detector_window[1::-1]-0.5
    ax.add_patch(mpl.patches.Rectangle(top_left_corner, detector_window[3], detector_window[2],
                                       color='r', fill=False))

def plot_sub_view_masks(ax, calcam_calib, image_coords='Display'):
    """Plot alpha layer showing calcam subview masks"""
    if image_coords.lower() == 'display':
        image_shape = calcam_calib.geometry.get_display_shape()
    else:
        image_shape = calcam_calib.geometry.get_original_shape()

    x_pix = np.arange(image_shape[0])
    y_pix = np.arange(image_shape[1])
    data = xr.Dataset(coords={'x_pix': x_pix, 'y_pix': y_pix})

    data['subview_mask_im'] = (('y_pix', 'x_pix'), calcam_calib.get_subview_mask(coords=image_coords))

    figure_xarray_imshow(data, key='subview_mask_im', ax=ax, alpha=0.3, add_colorbar=False, cmap='Pastel2',
                         axes_off=False,
                         show=False)

def plot_analysis_path(ax, xpix, ypix, xpix_out_of_frame=None, ypix_out_of_frame=None, **kwargs):
    """Plot pixel path over image"""
    kws = dict(marker='o', ms=2, ls='-', lw=1)
    kws.update(kwargs)
    ax.plot(xpix, ypix, **kws)
    if xpix_out_of_frame is not None:
        kws = dict(marker='x', ms=2, ls='-', lw=1, color='r')
        kws.update(kwargs)
        ax.plot(xpix_out_of_frame, ypix_out_of_frame, **kws)

def figure_spatial_res_max(data, ax=None, clip_range=[None, 15], log_cmap=True, aspect='equal', axes_off=True,
                           save_fn=None, show=False):
    key = 'spatial_res_max'
    cbar_label = 'Spatial resolution (max) [mm]'
    scale_factor = 1e3
    cmap = 'gray_r'

    figure_xarray_imshow(data, key, cbar_label=cbar_label, ax=ax, cmap=cmap, log_cmap=log_cmap, axes_off=axes_off,
                         aspect=aspect, clip_range=clip_range,
                         scale_factor=scale_factor, save_fn=save_fn, show=show)

def figure_spatial_res_x(data, ax=None, log_cmap=True, aspect='equal', axes_off=True, save_fn=None, show=False):
    key = 'spatial_res_x'
    cbar_label = 'Spatial resolution (x) [mm]'
    scale_factor = 1e3
    cmap = 'gray_r'
    figure_xarray_imshow(data, key, cbar_label=cbar_label, ax=ax, cmap=cmap, log_cmap=log_cmap, axes_off=axes_off, aspect=aspect,
                         scale_factor=scale_factor, save_fn=save_fn, show=show)

def figure_spatial_res_y(data, ax=None, log_cmap=True, aspect='equal', axes_off=True, save_fn=None, show=False):
    key = 'spatial_res_y'
    cbar_label = 'Spatial resolution (y) [mm]'
    scale_factor = 1e3
    cmap = 'gray_r'
    figure_xarray_imshow(data, key, cbar_label=cbar_label, ax=ax, cmap=cmap, log_cmap=log_cmap, axes_off=axes_off,
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

def plot_image_data_temporal_stats(data, key, ax=None, stats=('max', 'mean', 'median', 'min'), x_key='n',
                                   log_y=False, save_fn=None, show=False, **kwargs):
    from cycler import cycler
    fig, ax, ax_passed = get_fig_ax(ax, num=key)  # TODO: Custom axis , sharex=True, sharey=True

    data_plot = data[key]
    x = data_plot[x_key]

    cycler_props = (cycler(ls=['-', '--', '-.', ':']))
            # cycler(color=['c', 'm', 'y', 'k']) +
            #      cycler(lw=[1, 2, 3, 4]))
    # ax.set_prop_cycle(cycler_props)
    for stat, props in zip(stats, cycler_props):
        try:
            line_data = getattr(data_plot, stat)(dim=('x_pix', 'y_pix'))
        except AttributeError as e:
            raise e
        else:
            ax.plot(x, line_data, label=stat, alpha=0.6, **props)

    ax.set_xlabel(f'{x.attrs["long_name"]} [{x.attrs["units"]}]')
    ax.set_ylabel(f'{data_plot.attrs["long_name"]} [{data_plot.attrs["units"]}]')
    if log_y:
        ax.set_yscale('log')
    legend(ax)
    if not ax_passed:
        plt.tight_layout()
        ax.set_aspect('equal')
    if save_fn:
        plt.savefig(save_fn, bbox_inches='tight', transparent=True, dpi=dpi)
        logger.info(f'Saved y image histogram figure of {key} to: {save_fn}')
    if show:
        plt.show()

def animate_frame_data(data, key='frame_data', ax=None, duration=10, interval=None, cmap='gray',
                       axes_off=True, fig_kwargs=None, nth_frame='dynamic', save_kwargs=None,
                       save_path_fn=None, show=True, **kwargs):
    if fig_kwargs is None:
        fig_kwargs = {'num': key}
    # if save_path_fn is not None:
    #     save_path_fn = Path(save_path_fn)
    #     if save_path_fn.is_dir():
    #         fn = fn_pattern.format(pulse=pulse, key=key)
    frames = data[key]
    n_frames = len(frames)
    if nth_frame == 'dynamic':
        nth_frame = int(np.floor(np.max([1, (np.log10(n_frames)-1)**3])))
        logger.info(f'Animation frame step={nth_frame}')

    fig, ax, anim = animate_image_data(frames, ax=ax, duration=duration, interval=interval, cmap=cmap,
                                       axes_off=axes_off, fig_kwargs=fig_kwargs, nth_frame=nth_frame,
                                       save_path_fn=save_path_fn, show=show, **kwargs)
    return fig, ax, anim

def animate_image_data(frames, ax=None, duration=None, interval=None, cmap='viridis', axes_off=True, fig_kwargs=None,
                       nth_frame=1, save_kwargs=None, save_path_fn=None, show=True):
    # import numpy as np
    # import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # plt.ion()
    if fig_kwargs is None:
        fig_kwargs = {}

    nframes = int(len(frames) / nth_frame)
    frame_nos = np.arange(0, nframes, nth_frame, dtype=int)

    logger.info(f'Plotting matplotlib annimation ({nframes} frames)')

    fig, ax, ax_passed = get_fig_ax(ax=ax, **fig_kwargs)

    if interval is None:
        if (duration is None):  # duration fo whole movie/animation in s
            interval = 200  # ms
        else:
            interval = duration / nframes * 1000

    img_data = frames[0]
    img = ax.imshow(img_data, cmap=cmap)

    div = make_axes_locatable(ax)
    ax_cbar = div.append_axes('right', '5%', '5%')
    cbar = fig.colorbar(img, cax=ax_cbar)
    tx = ax.set_title(f'Frame 0/{nframes-1}')

    if axes_off:
        ax.set_axis_off()

    # xdata, ydata = [], []
    # ln, = plt.plot([], [], 'ro')

    def init():
        # ax.set_xlim(0, 2 * np.pi)
        # ax.set_ylim(-1, 1)
        # return img, cbar, tx
        # return ln,
        pass

    def update(frame_no):
        # xdata.append(frame)
        # ydata.append(np.sin(frame))
        # ln.set_data(xdata, ydata)
        frame = frames[frame_no]
        vmax = np.max(frame)
        vmin = np.min(frame)

        img.set_data(frame)
        img.set_clim(vmin, vmax)

        # levels = np.linspace(vmin, vmax, 200, endpoint=True)
        # cf = ax.contourf(frame, vmax=vmax, vmin=vmin, levels=levels)
        # ax_cbar.cla()
        # fig.colorbar(img, cax=ax_cbar)
        tx.set_text(f'Frame {frame_no}/{nframes-1}')
        # return img, cbar, tx
        # return ln,

    anim = FuncAnimation(fig, update, frames=frame_nos, #np.linspace(0, 2 * np.pi, 128),
                        interval=interval,
                        # init_func=init,
                        blit=False)
    if save_path_fn is not None:
        save_path_fn = str(Path(save_path_fn).resolve())
        try:
            kwargs = dict(fps=30)
            if save_kwargs is not None:
                kwargs.update(save_kwargs)
            anim.save(save_path_fn, writer='imagemagick', **kwargs)
                      # savefig_kwargs=dict(bbox_inches='tight', transparent=True))  # transparent makes blury
        except Exception as e:
            logger.exception(f'Failed to save matplotlib animation gif to {save_path_fn}')
        else:
            logger.info(f'Saved animation gif to {save_path_fn}')

    if show:
        plt.show()

    return fig, ax, anim


if __name__ == '__main__':
    nframes = 50
    frame = np.arange(320 * 256).reshape((320, 256))
    frames = np.zeros((nframes,) + frame.shape)
    frame_nos = np.arange(0, nframes, dtype=int)
    scale_factors = 1 + 0.25 * np.sin(1.5 * np.pi + 1.5 * np.pi * frame_nos / len(frame_nos))
    frames = scale_factors[:, np.newaxis, np.newaxis] * frame

    animate_image_data(frames)
    pass