#!/usr/bin/env python

"""


Created: 
"""

import logging
from copy import copy

import fire
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from cycler import cycler

from fire.misc.utils import make_iterable
from fire.misc import data_structures
from fire.plotting import plot_tools
from fire.plotting.plot_tools import get_fig_ax, color_shade

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

def figure_spatial_profile_1d(data_path, key, path_name='path0', x_coord=None, slice_=None, reduce=None,
                              offset_x_peak=False, offset_x_param=None, normalise_factor=False, ax=None,
                              plot_kwargs=None,
                              label=True, legend=True, meta=None, show=True, save_fn=None):
    """Plot profile of variable given by key along analysis path

      Args:
          data_path:
          key:
          slice_:
          ax:
          plot_kwargs:

      Returns: ax

    Args:
        data_path: Dataset containing plot data
        key: Name of variable in dataset to plot
        path_name:
        x_coord:
        slice_: (Temporal) slice to plot. Dict of {'coord': slice/value}
        reduce: Tuple to reduce dimensions: (coord, func, args) eg [('t', np.mean, ())]
        offset_x_peak: Apply shift to x axis to put peak value at 0. Use to align profiles at different radial locations
        offset_x_param:
        ax: Matplotlib axis to plot to
        plot_kwargs: Additional formatting args to pass to plt.plot
        label:
        legend:
        meta:
        show:
        save_fn:

    Returns:

    """
    from fire.physics.physics_parameters import add_custom_shifted_coord

    fig, ax, ax_passed = plot_tools.get_fig_ax(ax, num=f'{key} temporal profile')

    if (slice_ is None) and (reduce is None):
        slice_ = {'n': np.floor(np.median(data_path['n']))}

    kws = {'color': None}
    if isinstance(plot_kwargs, dict):
        kws.update(plot_kwargs)

    if path_name is not None:
        key = key.format(path=path_name)

    data_plot = data_path[key]
    # if (data_plot.ndim == 1):
    #     data_plot = data_plot.sel(slice_, method='nearest')
    x_slice = None

    if data_plot.ndim > 1:
        if slice_ is not None:
            # Take slice through 2D data for 1D plot
            slice_dim = list(slice_.keys())[0]
            if slice_dim not in data_plot.dims:
                data_plot = data_plot.swap_dims({data_plot.dims[0]: slice_dim})
            method = None if np.any([isinstance(v, slice) for v in slice_.values()]) else 'nearest'
            slice_ = copy(slice_)
            x_slice = {x_coord: slice_.pop(x_coord, slice(None, None))}
            data_plot = data_plot.sel(slice_, method=method)

        if reduce is not None:
            # Take average/max etc stat to reduce dimensionality
            for coord, func, args in reduce:
                # data_plot = xr.apply_ufunc(func, data_plot, *args,
                #                            input_core_dims=((coord,), ()), kwargs=dict(axis=axis_keep))
                data_plot = data_structures.reduce_2d_data_array(data_plot, func, coord, args)

    if x_coord is not None:
        data_plot = data_plot.swap_dims({data_plot.dims[0]: x_coord})

    if x_slice is not None:
        data_plot = data_plot.sortby(x_coord)
        data_plot = data_plot.sel(x_slice, method=None)

    if label not in (False, None):
        if label is True:
            label = data_plot.attrs.get('symbol', data_plot.name)
        label = label.replace(f'_{path_name}', '').replace('_', ' ').replace('r peak', r'$R_{peak}$')
        kws['label'] = label

    if offset_x_peak:
        from fire.physics.physics_parameters import add_peak_shifted_coord
        data_plot, coord_new = add_peak_shifted_coord(data_plot)

    if offset_x_param is not None:
        from fire.physics.physics_parameters import add_custom_shifted_coord
        data_plot, coord_new = add_custom_shifted_coord(data_plot, x_coord, 'shifted_coord', data_path,
                                           offset_param=offset_x_param, slice_=slice_)

    if normalise_factor not in (False, None):
        if normalise_factor is True:
            normalise_factor = 1  # data_plot.abs().max()
        data_plot = normalise_factor * data_plot / data_plot.abs().max()
        # TODO: update label to include normalisation factor

    data_plot.attrs.update(data_path[key].attrs)

    artist = data_plot.plot.line(ax=ax, **kws)
    ax.title.set_fontsize(10)

    plot_tools.legend(ax, legend=legend, only_multiple_artists=True)
    plot_tools.show_if(show=show, close_all=False)
    plot_tools.save_fig(save_fn, fig=fig, save=(save_fn is not None))

    return fig, ax, data_plot, artist

def figure_poloidal_cross_section(image_data=None, path_data=None, path_names='path0', ax=None,
                                  closest_points=True, tile_edges=True, legend=True, axes_off=False,
                                  color_image='orange', color_paths='green',
                                  pulse=50000, no_cal=False, pupil_coords=None, show=True):
    import warnings
    from fire.geometry.s_coordinate import get_nearest_boundary_coordinates
    if fire.active_machine_plugin is not None:
        (machine_plugins, machine_plugins_info) = fire.active_machine_plugin
        plugin_module = machine_plugins_info['module']
        plot_vessel_outline = getattr(plugin_module, 'plot_vessel_outline')
        get_wall_rz_coords = getattr(plugin_module, 'get_wall_rz_coords')
    else:
        (machine_plugins, machine_plugins_info) = {}, {}
        from fire.plugins.machine_plugins.mast_u import plot_vessel_outline_mastu as plot_vessel_outline
        from fire.plugins.machine_plugins.mast_u import get_wall_rz_coords

    path_names = make_iterable(path_names)
    fig, ax, ax_passed = get_fig_ax(ax, num='pol cross section')

    if image_data is not None:
        r_im, z_im = np.array(image_data['R_im']).flatten(), np.array(image_data['z_im']).flatten()
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
    if (tile_edges is True):
        # TODO: Generalise get/plot tile edges functions
        if (machine_plugins.get('machine') == 'MAST-U'):
            from fire.plugins.machine_plugins.mast_u import plot_poloidal_tile_edges_mastu
            plot_poloidal_tile_edges_mastu(ax=ax, top=top, bottom=bottom, markersize=4, color='k', label='Tile edges', show=False)

    # TODO: Plot rays from camera pupil showing field of view

    if image_data is not None:
        # TODO: SPlit into functions plot_rz_points and plot_nearest_boundary_coordinates
        # Plot the poloidal cross section of all the surfaces visible in the images
        color = color_image if (not closest_points) else color_shade(color_image, -30)
        ax.plot(r_im, z_im, ls='', marker='o', markersize=2, color=color, label='Image pixels')
        if closest_points:
            r_wall, z_wall = get_wall_rz_coords(no_cal=no_cal, shot=pulse, ds=1e-3)  # Get at reduced spatial res
            closest_coords, closest_dist, closest_index = get_nearest_boundary_coordinates(r_im, z_im, r_wall, z_wall)
            r_close, z_close = closest_coords[:, 0], closest_coords[:, 1]
            ax.plot(r_close, z_close, ls='', marker='o', markersize=1, color=color_image, alpha=0.6,
                    label='Wall closest to image pixels')
            logger.info(f'Dists for all image coords: min: {np.nanmin(closest_dist):0.3g}, '
                                              f'mean: {np.nanmean(closest_dist):0.3g}, '
                                              f'max: {np.nanmax(closest_dist):0.3g}')

    if path_data is not None:
        # Plot the poloidal cross section of the surfaces along the analysis path(s)
        for path, plot_args in zip(path_names, cycler(color=color_paths)):
            r_path, z_path = np.array(path_data[f'R_{path}']), np.array(path_data[f'z_{path}'])
            color = plot_args['color']
            color = color if (not closest_points) else color_shade(color, -30)
            # ax.plot(r_path, z_path, ls='', marker='o', markersize=2, label='Analysis path pixels', alpha=0.3,
            #         **plot_args)
            if closest_points:
                r_wall, z_wall = get_wall_rz_coords(no_cal=no_cal, shot=pulse, ds=1e-4)
                closest_coords, closest_dist, closest_index = get_nearest_boundary_coordinates(r_path, z_path, r_wall, z_wall)
                r_close, z_close = closest_coords[:, 0], closest_coords[:, 1]
                ax.plot(r_close, z_close, ls='', marker='o', markersize=2, color=color_paths, alpha=0.9,
                        label=f'Wall closest to analysis {path} pixels')
                logger.info(f'Dists for {path} coords: min: {np.nanmin(closest_dist):0.3g}, '
                                                f'mean: {np.nanmean(closest_dist):0.3g}, '
                                                f'max: {np.nanmax(closest_dist):0.3g}')
            pass

    if pupil_coords is not None:
        # pupil_coords_toroidal = cartesian_to_toroidal(*tuple(pupil_coords), angles_in_deg=True)
        r_pupil = np.hypot(*pupil_coords[:2])
        ax.plot(r_pupil, pupil_coords[2], marker='8', color='k', ms=6, alpha=0.6, label='camera pupil')

    kws = {'fontsize': 7, 'framealpha': 0.7, 'facecolor': 'white', 'fancybox': True}
    plot_tools.legend(ax, legend=legend, **kws)

    plot_tools.show_if(show=show, tight_layout=True)

def figure_top_down(path_data=None, image_data=None, path_names='path0', ax=None, machine_plugins=None,
                    phi_labels=True, legend=True, axes_off=False, image_color='orange', path_colors='green',
                    pupil_coords=None, show=True):
    from cycler import cycler
    path_names = make_iterable(path_names)
    fig, ax, ax_passed = get_fig_ax(ax, num='top down view')

    if image_data is not None:
        # Plot the top down view of all the surfaces visible in the images
        x_im, y_im = np.array(image_data[f'x_im']), np.array(image_data[f'y_im'])
        ax.plot(x_im, y_im, ls='', marker='o', markersize=2, color=image_color, label='Image pixels')

    if path_data is not None:
        # Plot the top down view of the analysis path(s)
        for path, plot_args in zip(path_names, cycler(color=path_colors)):
            x_path, y_path = np.array(path_data[f'x_{path}']), np.array(path_data[f'y_{path}'])
            ax.plot(x_path, y_path, ls='', marker='o', markersize=2, label=f'Analysis path "{path}" pixels',
                    alpha=0.7, **plot_args)

    if (machine_plugins is not None) and ('plot_vessel_top_down' in machine_plugins):
        try:
            machine_plugins['plot_vessel_top_down'](ax=ax, axes_off=axes_off, phi_labels=phi_labels)
        except Exception as e:
            raise e  # Fails for MAST, JET due to no wall definition etc

    if pupil_coords is not None:
        ax.plot(pupil_coords[0], pupil_coords[1], marker='8', color='k', ms=6, alpha=0.6, label='camera pupil')

    kws = {'fontsize': 7, 'framealpha': 0.7, 'facecolor': 'white', 'fancybox': True}
    plot_tools.legend(ax, legend=legend, **kws)

    plot_tools.show_if(show=show, tight_layout=True)

    return fig, ax

if __name__ == '__main__':
    pass
