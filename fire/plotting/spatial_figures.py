#!/usr/bin/env python

"""


Created: 
"""

import logging
from typing import Union, Iterable, Sequence, Tuple, Optional, Any, Dict
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from fire.misc.utils import make_iterable
from fire.plotting.plot_tools import get_fig_ax, repeat_color, color_shade
from matplotlib import pyplot as plt
from cycler import cycler

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

def figure_poloidal_cross_section(image_data=None, path_data=None, path_names='path0', ax=None,
                                  closest_points=True, tile_edges=True, legend=True, axes_off=False,
                                  color_image='orange', color_paths='green',
                                  pulse=50000, no_cal=False, show=True):
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
            from fire.plugins.machine_plugins.mast_u import plot_tile_edges_mastu
            plot_tile_edges_mastu(ax=ax, top=top, bottom=bottom, markersize=4, color='k', label='Tile edges', show=False)

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
            print(f'Dists for all image coords: min: {np.nanmin(closest_dist):0.3g}, '
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
                print(f'Dists for {path} coords: min: {np.nanmin(closest_dist):0.3g}, '
                                                f'mean: {np.nanmean(closest_dist):0.3g}, '
                                                f'max: {np.nanmax(closest_dist):0.3g}')
            pass
    if legend:
        kws = {'fontsize': 7, 'framealpha': 0.7, 'facecolor': 'white', 'fancybox': True}
        leg = ax.legend(**kws)
        leg.set_draggable(True)


    if show:
        # plt.tight_layout()
        plt.show()


def figure_top_down(path_data=None, image_data=None, path_names='path0', ax=None, machine_plugins=None,
                    phi_labels=True, legend=True, axes_off=False, image_color='orange', path_colors='green',
                                  show=True):
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

    if legend:
        kws = {'fontsize': 7, 'framealpha': 0.7, 'facecolor': 'white', 'fancybox': True}
        leg = ax.legend(**kws)
        leg.set_draggable(True)

    if show:
        # plt.tight_layout()
        plt.show()

    return fig, ax

if __name__ == '__main__':
    pass
