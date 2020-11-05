#!/usr/bin/env python

"""


Created: 
"""

import logging, os
from typing import Union, Iterable, Sequence, Tuple, Optional, Any, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt

from fire.misc.utils import mkdir

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


def get_fig_ax(ax=None, num=None, fig_shape=(1, 1), dimensions=2, **kwargs):
    """If passed None for the ax keyword, return new figure and axes

    Args:
        ax: Existing axis instance to return figure instance for. If None, a new figure and axis is returned.
        num: Name of new figure window (if ax=None)
        fig_shape: Dimensions of new figure window (if ax=None)
        dimensions: Number of dimensions axes for axis (2/3)
        **kwargs: Additional kwargs passed to plt.subplots

    Returns: fig (figure instance), ax (axis instance), ax_passed (bool specifying if ax was passed or generated here)

    """

    if ax is None:
        if dimensions == 3:
            if np.sum(fig_shape) > 2:
                raise NotImplementedError
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
            fig = plt.figure(num=num)
            ax = fig.add_subplot(1, 1, 1, projection='3d')
        else:
            fig, ax = plt.subplots(*fig_shape, num=num, constrained_layout=True, **kwargs)
        ax_passed = False
    else:
        fig = ax.figure
        ax_passed = True
    return fig, ax, ax_passed

def annotate_axis(ax, string, x=0.85, y=0.955, fontsize=16, coords='axis',
                  bbox=(('facecolor', 'w'), ('ec', None), ('lw', 0), ('alpha', 0.5), ('boxstyle', 'round')),
                  horizontalalignment='center', verticalalignment='center', multialignment='center', **kwargs):
    if isinstance(bbox, (tuple, list)):
        bbox = dict(bbox)
    elif isinstance(bbox, dict):
        bbox_user = bbox
        bbox = dict((('facecolor', 'w'), ('ec', None), ('lw', 0), ('alpha', 0.5), ('boxstyle', 'round')))
        bbox.update(bbox_user)

    if coords == 'axis':
        transform = ax.transAxes
    elif coords == 'data':
        transform = ax.transData
    ax.text(x, y, string, fontsize=fontsize, bbox=bbox, transform=transform,
            horizontalalignment=horizontalalignment, verticalalignment=verticalalignment, multialignment=multialignment,
            **kwargs)

def legend(ax, handles=None, labels=None, legend=True, only_multiple_artists=True, zorder=None, **kwargs):
    """Finalise legends of each axes"""
    kws = {'fontsize': 14, 'framealpha': 0.7, 'facecolor': 'white', 'fancybox': True}
    leg = None
    try:
        handles_current, labels_current = ax.get_legend_handles_labels()
        # Only produce legend if more than one  artist has a label
        if (not only_multiple_artists) or (len(handles_current) > 1) or (handles is not None):
            args = () if handles is None else (handles, labels)
            kws.update(kwargs)
            leg = ax.legend(*args, **kws)
            leg.set_draggable(True)
            if zorder is not None:
                leg.set_zorder(zorder)
    except ValueError as e:
        #  https: // github.com / matplotlib / matplotlib / issues / 10053
        logger.error('Not sure how to avoid this error: {}'.format(e))
    if not legend:
        leg = ax.legend()
        leg.remove()
    return leg

def save_fig(path_fn, fig=None, path=None, transparent=True, bbox_inches='tight', dpi=90, save=True,
             image_formats=None, image_format_subdirs='subsequent',
             mkdir_depth=None, mkdir_start=None, description='', verbose=True):
    if not save:
        return False
    if fig is None:
        fig = plt.gcf()
    if path is not None:
        path_fn = os.path.join(path, path_fn)
    path_fn = os.path.realpath(os.path.expanduser(path_fn))
    # if not pos_path(path_fn, allow_relative=True):  # string path
    #     raise IOError('Not valid save path: {}'.format(path_fn))

    if (mkdir_depth is not None) or (mkdir_start is not None):
        mkdir(os.path.dirname(path_fn), depth=mkdir_depth, start_dir=mkdir_start)

    if image_formats is None:
        _, ext = os.path.splitext(path_fn)
        path_fns = {ext: path_fn}
    else:
        # Handle filesnames without extension with periods in
        path_fn0, ext = os.path.splitext(path_fn)
        path_fn0 = path_fn0 if len(ext) <= 4 else path_fn
        path_fns = {}
        for i, ext in enumerate(image_formats):
            path_fn = '{}.{}'.format(path_fn0, ext)
            if ((image_format_subdirs == 'all') or (image_format_subdirs is True) or
                    ((image_format_subdirs == 'subsequent') and (i > 0))):
                path_fn = insert_subdir_in_path(path_fn, ext, -1, create_dir=True)
            path_fns[ext] = path_fn
    for ext, path_fn in path_fns.items():
        try:
            fig.savefig(path_fn, bbox_inches=bbox_inches, transparent=transparent, dpi=dpi)
        except RuntimeError as e:
            logger.exception('Failed to save plot to: {}'.format(path_fn))
            raise e
    if verbose:
        logger.info('Saved {} plot to:\n{}'.format(description, path_fns))
        print('Saved {} plot to:'.format(description))
        print(path_fns)

def get_previous_artist_color(ax=None, artist_ranking=('line', 'pathcollection'), artist_ranking_str=None):
    artist_type_options = ('line', 'pathcollection')
    if ax is None:
        ax = plt.gca()
    if (artist_ranking_str is not None) and any([a in artist_ranking_str for a in artist_type_options]):
        artist_ranking = [a for a in artist_type_options if a in artist_ranking_str]

    for artist_type in artist_ranking:
        assert artist_type in artist_type_options
        if artist_type == 'line' and len(ax.lines) != 0:
            artist = ax.lines[-1]
            color = artist.get_color()
            break
        elif artist_type == 'pathcollection' and len(ax.collections) != 0:
            artists = [a for a in ax.collections if isinstance(a, matplotlib.collections.PathCollection)]
            if len(artists) > 0:
                artist = artists[-1]
                color = artist.get_facecolor()[0]  # [:2]
                break
    else:
        logger.warning("Can't repeat line color - no previous lines or path collections on axis")
        return 'k'
    return color

def get_previous_line_color(ax=None):
    """Return color of previous line plotted to axis"""
    if ax is None:
        ax = plt.gca()
    if len(ax.lines) != 0:
        color = ax.lines[-1].get_color()
    else:
        logger.warning("Can't repeat line color - no previous lines or path collections on axis")
        return 'k'

    return color

def color_shade(color, percentage):
    """Crude implementation to make color darker or lighter until matplotlib function is available:
    https://github.com/matplotlib/matplotlib/pull/8895"""
    from matplotlib import colors as mcolors
    c = mcolors.ColorConverter().to_rgb(color)
    c = np.clip(np.array(c)+percentage/100., 0, 1)
    return c

def repeat_color(ax=None, shade_percentage=None, artist_string=None):
    if ax is None:
        ax = plt.gca()
    color = get_previous_artist_color(ax, artist_ranking_str=artist_string)
    if (shade_percentage is None) and (artist_string is not None):
        if '+' in artist_string or '-' in artist_string:
            shade_percentage = float(artist_string.split('+')[-1].split('-')[-1])
    if shade_percentage is not None:
        c = color_shade(color, shade_percentage)
    return c

def format_poloidal_plane_ax(ax, units='m'):
    ax.set_xlabel(f'R [{units}]')
    ax.set_ylabel(f'Z [{units}]')
    ax.set_aspect('equal')
    return ax

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

def plot_vessel_outline(r, z, ax=None, top=True, bottom=True, s_start_coord=(0,0), aspect='equal', ax_labels=True,
                              axes_off=False, show=True, **kwargs):
    import matplotlib.pyplot as plt
    from fire.geometry.s_coordinate import separate_rz_points_top_bottom
    if ax is None:
        fig, ax = create_poloidal_cross_section_figure(1, 1)
    else:
        fig = ax.figure

    (r_bottom, z_bottom), (r_top, z_top) = separate_rz_points_top_bottom(r, z, top_coord_start=s_start_coord,
                                                                         bottom_coord_start=s_start_coord)
    # ax.plot(r0, z0, marker='x', ls='-')
    if bottom:
        ax.plot(r_bottom, z_bottom, marker='', ls='-', **kwargs)
    if top:
        ax.plot(r_top, z_top, marker='', ls='-', **kwargs)

    ax.set_aspect(aspect)
    if axes_off:
        ax.set_axis_off()
    # elif ax_labels:  # Already done in create_poloidal_cross_section_figure()
    #     ax.set_xlabel('$R$ [m]')
    #     ax.set_ylabel('$z$ [m]')

    if show:
        plt.tight_layout()
        plt.show()
    return fig, ax

def plot_vessel_top_down(surface_radii, keys_plot, ax=None, axes_off=False, phi_labels=True, keys_plot_strong=()):
    from matplotlib import patches
    from fire.plotting.plot_tools import get_fig_ax
    from fire.geometry.geometry import cylindrical_to_cartesian
    fig, ax, ax_passed = get_fig_ax(ax)

    r_wall = surface_radii['R_wall']
    n_sectors = surface_radii['n_sectors']

    # Plot vessel
    wall = patches.Circle((0, 0), radius=r_wall, facecolor='b', edgecolor='k', alpha=0.1)
    ax.add_patch(wall)

    # Plot tile radii etc
    for key in keys_plot:
        r = surface_radii[key]
        alpha = 0.6 if key in keys_plot_strong else 0.3
        wall = patches.Circle((0, 0), radius=r, facecolor=None, fill=False, edgecolor='k', ls='--', alpha=alpha)
        ax.add_patch(wall)

    # Lines between sectors
    for i in np.arange(n_sectors):
        x, y = cylindrical_to_cartesian(r_wall, i*360/n_sectors, angles_units='degrees')
        ax.plot([0, x], [0, y], ls='--', c='k', lw=1)

    # Sector numbers
    for i in np.arange(n_sectors):
        x, y = cylindrical_to_cartesian(r_wall*0.9, 90-(i+0.5)*360/n_sectors, angles_units='degrees')
        ax.text(x, y, f'{i+1}', horizontalalignment='center', verticalalignment='center')

    # Phi labels
    if phi_labels:
        for i in np.arange(4):
            phi = i*360/4
            x, y = cylindrical_to_cartesian(r_wall, phi, angles_units='degrees')
            label = f'$\phi={phi:0.0f}^\circ$'
            # annotate_axis(ax, label, x, y, fontsize=16, color='k')
            ax.text(x, y, label, horizontalalignment='left', verticalalignment='bottom')
        ax_scale = 1.2
    else:
        ax_scale = 1.1

    ax.set_aspect(1)
    ax.set_xlim(-r_wall * ax_scale, r_wall * ax_scale)
    ax.set_ylim(-r_wall * ax_scale, r_wall * ax_scale)
    if axes_off:
        ax.set_axis_off()
    else:
        ax.set_xlabel(r'x [m]')
        ax.set_ylabel(r'y [m]')
        # plt.tight_layout()

    return fig, ax


if __name__ == '__main__':
    pass