# -*- coding: future_fstrings -*-
"""MAST specific plugin functions for FIRE.

Functions and top level variables in this file with appropriate names will be imported and used by FIRE to provide
functionality and information specific to this machine.

The most important role of the machine plugin is to provide a means of looking up the tile surface 's' coordinate
against which heat flux profiles etc are defined. Additional machine specific information and tools include
identifying sector numbers and other location based labels and plotting methods for visualising this information.

Author: Tom Farley (tom.farley@ukaea.uk)
Created: 01-2020
"""
import logging

import numpy as np
from fire.plotting.plot_tools import format_poloidal_plane_ax
from fire.geometry.s_coordinate import interpolate_rz_coords, separate_rz_points_top_bottom, calc_s_coord_lookup_table, \
    get_nearest_s_coordinates, get_nearest_rz_coordinates, get_nearest_boundary_coordinates

from fire.geometry.geometry import cartesian_to_toroidal
from fire.misc.utils import make_iterable

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# ===================== PLUGIN MODULE ATTRIBUTES =====================
# Which plugin attributes/functions are required or optional is set in your fire_config.json file under "plugins".

# Required: Name of plugin module (typically name of machine == name of file), needed to be located as a plugin module
machine_plugin_name = 'mast_u'

# Recommended
machine_name = 'MAST-U'  # Will be cast to lower case (and '-' -> '_') in FIRE
plugin_info = {'description': 'This plugin supplies functions for MAST-U specific operations/information'}  # exta info
location_labels_im = ['sector', 's_global']  # Parameters used to label coordinates across the whole image

# Optional/other
n_sectors = 12  # Used internally in funcs below
# Machine specific
None

# Boxes to pass to fire.s_coordinate.remove_false_rz_surfaces
false_rz_surface_boxes_default = [((1.512, 1.6), (-0.81789, 0.81789)),]
# Origin for s coordinates/divider between top and bottom of machine
s_start_coord_default = (0.260841, 0)

# Use same plugin funcs for machine sector and s_path coordinate as for MAST
from fire.plugins.machine_plugins.mast import (get_machine_sector, get_tile_louvre_index, get_tile_louvre_label,
                                               get_s_coord_path)
# See bottom of file for function aliases
# ====================================================================

module_default = object()

# Radii of edges of different structures for top down view plot etc
surface_radii = {'R_wall': 2.0,
                'R_HL04': 2.1333,
                'R_T1': 0.333,
                'R_T2': 0.540,
                'R_T3': 0.905,
                'R_T4': 1.081,
                'R_T5': 1.379,
                'R_T5_top': 1.746,
                'n_sectors': 12,
                }

def get_wall_rz_coords(no_cal=True, signal="/limiter/efit", shot=50000, ds=None):
    """Return (R, Z) coordinates of points defining wall outline of MAST-U tile surfaces

    This is normally safe to call with default arguments.

    Args:
        no_cal: Whether to return idealised CAD coordinates without spatial calibration corrections
        signal: UDA signal for wall coords
        shot: Shot number to get wall definition for

    Returns: Tuple of (R, Z) coordinate arrays

    """
    import pyuda
    # TODO: Enable returning wall for MAST ie shot < 50000
    client=pyuda.Client()
    wall_data = client.geometry(signal, shot, no_cal=no_cal)
    r = wall_data.data.R
    z = wall_data.data.Z
    if ds is not None:
        r, z = interpolate_rz_coords(r, z, ds=ds, false_surface_boxes=false_rz_surface_boxes_default)
    return r, z

def get_tile_edge_coords_mastu(no_cal=True, signal="/limiter/efit", shot=50000, subset=True):
    """Return (R,Z) coords of main tile boundaries

    Args:
        no_cal: Whether to use idealised CAD coordinates without spatial calibration corrections
        signal: UDA signal for wall coords
        shot: Shot number to get wall definition for

    Returns: Tuple of (R, Z) coordinate arrays

    """
    min_tile_size = 0.10
    r0, z0 = get_wall_rz_coords(signal=signal, no_cal=no_cal, shot=shot)
    diff = np.linalg.norm([np.diff(r0), np.diff(z0)], axis=0)
    mask = (diff > min_tile_size) | (np.roll(diff, -1) > min_tile_size)
    mask = np.concatenate([[np.True_], mask])
    r_tiles, z_tiles = r0[mask], z0[mask]
    if subset:
        if subset is True:
            # Just pick out boundaries for tiles T1-T5
            n = len(r_tiles)
            subset = np.concatenate([np.arange(1, 9), np.arange(n-34, n-26)])
        r_tiles, z_tiles =  r_tiles[subset], z_tiles[subset]
    return r_tiles, z_tiles

def get_s_coords_tables_mastu(ds=1e-4, no_cal=True, signal="/limiter/efit", shot=50000):
    """Return dict of dataframes containing (R, Z, s) coordinates for top and bottom regions of the machine

    Args:
        ds: Resolution to interpolate wall coordinate spacing to in meters
        no_cal: Whether to use idealised CAD coordinates without spatial calibration corrections
        signal: UDA signal for wall coords
        shot: Shot number to get wall definition for

    Returns: Dict of dataframes containing (R, Z, s) coordinates for top and bottom regions of the machine

    """
    r0, z0 = get_wall_rz_coords(no_cal=no_cal, signal=signal, shot=shot)
    r, z = interpolate_rz_coords(r0, z0, ds=ds)
    (r_bottom, z_bottom), (r_top, z_top) = separate_rz_points_top_bottom(r, z, prepend_start_coord=True,
                                                                         bottom_coord_start=s_start_coord_default,
                                                                         top_coord_start=s_start_coord_default)

    s_bottom = calc_s_coord_lookup_table(r_bottom, z_bottom)
    s_top = calc_s_coord_lookup_table(r_top, z_top)

    s = {'s_bottom': s_bottom, 's_top': s_top}
    # s_bottom = s_bottom.to_xarray().rename({'s': 's_bottom'})
    # s_top = s_top.to_xarray().rename({'s': 's_top'})
    # s = xr.merge([s_bottom, s_top])
    return s

# def get_s_coord_table_for_point():

# TODO: Import from mastcodes/uda/python/mast/geom/geomTileSurfaceUtils.py ? - from mast.geom.geomTileSurfaceUtils

def get_nearest_s_coordinates_mastu(r, z, tol=5e-3, ds=1e-3, no_cal=True, signal="/limiter/efit", shot=50000):
    """Return closest tile surface 's' coordinates for supplied (R, Z) coordinates

    Args:
        r: Array of radial R coordinates
        z: Array of vertical Z coordinates
        tol: Tolerance distance for points from wall - return nans if further away than tolerance.
             5mm is appropriate for most surfaces, but should be increased to 25mm+ for T5 to account for ripple shaping
        ds: Resolution to interpolate wall coordinate spacing to in meters
        no_cal: Whether to use idealised CAD coordinates without spatial calibration corrections
        signal: UDA signal for wall coords
        shot: Shot number to get wall definition for

    Returns: Dict of s coordinates for top/bottom, (Array of 1/-1s for top/bottom of machine, Dict keying 1/-1 to s
             keys)

    """
    r, z = make_iterable(r, ndarray=True), make_iterable(z, ndarray=True)
    s_lookup = get_s_coords_tables_mastu(ds=ds, no_cal=no_cal, signal=signal, shot=shot)
    z_mask = (z <= 0)
    s = np.full_like(r, np.nan, dtype=float)
    position = np.full_like(r, np.nan, dtype=float)
    table_key = {-1: 's_bottom', 1: 's_top'}
    for mask, key, pos in zip([z_mask, ~z_mask], ['s_bottom', 's_top'], [-1, 1]):
        lookup_table = s_lookup[key]
        if np.any(mask) and (not np.all(np.isnan(r[mask]))):
            r_masked, z_masked = r[mask], z[mask]
            r_wall, z_wall, s_wall = lookup_table['R'], lookup_table['Z'], lookup_table['s']
            s[mask] = get_nearest_s_coordinates(r_masked, z_masked, r_wall, z_wall, s_wall, tol=tol)
            position[mask] = pos
    return s, (position, table_key)

# def create_poloidal_cross_section_figure(nrow=1, ncol=1, cross_sec_axes=((0, 0),)):
#     fig, axes = plt.subplots(nrow, ncol)
#     if nrow==1 and ncol==1:
#         format_poloidal_plane_ax(axes)
#     else:
#         for ax_coord in cross_sec_axes:
#             ax = axes[slice(*ax_coord)]
#             format_poloidal_plane_ax(ax)
#     plt.tight_layout()
#     return fig, axes

def get_s_coord_global(x_im, y_im, z_im, **kwargs):
    """Return MAST-U tile s coordinates for all pixels in image.

    This MAST-U 's' coordinate starts at 0m mid-way up the centre column and increases along tile surfaces to a maximum
    value of 6.088473 m (top) and 6.088573 m (bottom).
    This 's' coordinate is considered 'global' as it is predefined for all (R, Z) surfaces as apposed to a 'local' s
    starting at 0m along a specific path.

    Args:
        x_im        : x coordinates of each pixel in image
        y_im        : y coordinates of each pixel in image
        z_im        : z coordinates of each pixel in image
        **kwargs    : Arguments passed to get_nearest_s_coordinates_mastu

    Returns: MAST-U tile 's' coordinate for each pixel in the image

    """
    x, y, z = (np.array(d).flatten() for d in (x_im, y_im, z_im))
    r = np.linalg.norm([x, y], axis=0)
    # phi = np.arctan2(y_im, x_im)
    s, (position, table_key) = get_nearest_s_coordinates_mastu(r, z, **kwargs)
    s_im = s.reshape(x_im.shape)
    # TODO: Check if masking with nans is required in any situations
    return s_im

def format_coord(coord, **kwargs):
    """
    MAST-U coordinate formatter, includes sector number.

    Args:
        coord: Array of (x, y, z) coordinates describing a point in the machine

    Returns: String describing the position of the point within the machine

    """
    x, y, z = coord[0], coord[1], coord[2]
    r, phi = cartesian_to_toroidal(x, y)

    sector = get_machine_sector(phi)

    formatted_coord = 'X,Y,Z: ( {:.3f} m , {:.3f} m , {:.3f} m )'.format(x, y, z)
    formatted_coord = formatted_coord + u'\nR,Z,\u03d5: ( {:.3f} m , {:.3f}m , {:.1f}\xb0 )'.format(r, z, phi)
    formatted_coord = formatted_coord + '\n Sector {:.0f}'.format(sector)

    return formatted_coord


def plot_tile_edges_mastu(ax=None, top=True, bottom=True, show=True, **kwargs):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = create_poloidal_cross_section_figure(1, 1)
    else:
        fig = ax.figure

    r0, z0 = get_tile_edge_coords_mastu(no_cal=True, shot=50000)

    mask = np.ones_like(r0, dtype=bool)
    mask = mask * (z0 > 0) if not bottom else mask
    mask = mask * (z0 < 0) if not top else mask

    ax.plot(r0[mask], z0[mask], marker='x', ls='', **kwargs)

    if show:
        plt.show()
    return fig, ax

def plot_vessel_outline(ax=None, top=True, bottom=True, shot=50000, no_cal=False, aspect='equal', ax_labels=True,
                              axes_off=False, show=True, **kwargs):
    """ Plot vessel outline for MAST-U

    Args:
        ax:
        top:
        bottom:
        shot:
        no_cal:
        aspect:
        ax_labels:
        axes_off:
        show:
        **kwargs:

    Returns:

    """
    from fire.plotting.plot_tools import plot_vessel_outline

    r0, z0 = get_wall_rz_coords(no_cal=no_cal, shot=shot)

    fig, ax = plot_vessel_outline(r0, z0, ax=ax, top=top, bottom=bottom, aspect=aspect, ax_labels=ax_labels,
                                axes_off=axes_off, show=show)

    return fig, ax

def plot_vessel_top_down(ax=None, keys_plot=('R_T1', 'R_T2', 'R_T3', 'R_T4', 'R_T5', 'R_T5_top', 'R_HL04'),
                         keys_plot_strong=('R_T1', 'R_T4', 'R_T5', 'R_T5_top'),
            axes_off=False, phi_labels=True):

    from fire.plotting.plot_tools import plot_vessel_top_down

    fig, ax = plot_vessel_top_down(surface_radii, keys_plot, ax=ax, axes_off=axes_off, phi_labels=phi_labels,
                                   keys_plot_strong=keys_plot_strong)

    # from matplotlib import patches
    # from fire.plotting.plot_tools import get_fig_ax
    # from fire.geometry.geometry import cylindrical_to_cartesian
    # fig, ax, ax_passed = get_fig_ax(ax)
    #
    # r_wall = surface_radii['R_wall']
    # n_sectors = surface_radii['n_sectors']
    #
    # # Plot vessel
    # wall = patches.Circle((0, 0), radius=r_wall, facecolor='b', edgecolor='k', alpha=0.1)
    # ax.add_patch(wall)
    #
    # # Plot tile radii etc
    # for key in ['R_T1', 'R_T2', 'R_T3', 'R_T4', 'R_T5', 'R_T5_top', 'R_HL04']:
    #     r = surface_radii[key]
    #     alpha = 0.6 if key in ['R_T1', 'R_T4', 'R_T5', 'R_T5_top'] else 0.3
    #     wall = patches.Circle((0, 0), radius=r, facecolor=None, fill=False, edgecolor='k', ls='--', alpha=alpha)
    #     ax.add_patch(wall)
    #
    # # Lines between sectors
    # for i in np.arange(n_sectors):
    #     x, y = cylindrical_to_cartesian(r_wall, i*360/n_sectors, angles_units='degrees')
    #     ax.plot([0, x], [0, y], ls='--', c='k', lw=1)
    #
    # # Sector numbers
    # for i in np.arange(n_sectors):
    #     x, y = cylindrical_to_cartesian(r_wall*0.9, 90-(i+0.5)*360/n_sectors, angles_units='degrees')
    #     ax.text(x, y, f'{i+1}', horizontalalignment='center', verticalalignment='center')
    #
    # # Phi labels
    # if phi_labels:
    #     for i in np.arange(4):
    #         phi = i*360/4
    #         x, y = cylindrical_to_cartesian(r_wall, phi, angles_units='degrees')
    #         label = f'$\phi={phi:0.0f}^\circ$'
    #         # annotate_axis(ax, label, x, y, fontsize=16, color='k')
    #         ax.text(x, y, label, horizontalalignment='left', verticalalignment='bottom')
    #     ax_scale = 1.2
    # else:
    #     ax_scale = 1.1
    #
    # ax.set_aspect(1)
    # ax.set_xlim(-r_wall * ax_scale, r_wall * ax_scale)
    # ax.set_ylim(-r_wall * ax_scale, r_wall * ax_scale)
    # if axes_off:
    #     ax.set_axis_off()
    # else:
    #     ax.set_xlabel(r'x [m]')
    #     ax.set_ylabel(r'y [m]')
    #     # plt.tight_layout()

    return fig, ax

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from fire.plotting.plot_tools import create_poloidal_cross_section_figure
    fig, ax = create_poloidal_cross_section_figure()
    plot_tile_edges_mastu(ax=ax, show=True)
    ds = 1e-4
    # ds = 1e-2
    r0, z0 = get_wall_rz_coords(no_cal=True, shot=50000)
    r, z = interpolate_rz_coords(r0, z0, ds=ds, false_surface_boxes=false_rz_surface_boxes_default)
    print(f'Number of interpolated wall points: {len(r)}')
    s_tables = get_s_coords_tables_mastu()
    # point = (1.1, 0.9)
    # point = (1, 1)
    point = (1.1, -1.8)
    s_close, pos, table_key = get_nearest_s_coordinates_mastu(*point)
    s_close_table = s_tables[table_key[pos[0]]]
    wall_s_close = get_nearest_rz_coordinates(s_close, s_close_table['R'].values, s_close_table['Z'].values,
                                              s_close_table['s'].values)
    wall_close = get_nearest_boundary_coordinates(*point, r_boundary=r, z_boundary=z)[0]
    fig, ax = plt.subplots(1, 1)
    ax.plot(r, z, marker='x', ls='')
    ax.plot(r0, z0, marker='x', ls='')
    # ax.plot(r0[:10], z0[:10], marker='x', ls='')
    ax.plot(point[0], point[1], marker='x', ls='', c='r', label='point')
    ax.plot(wall_close[0], wall_close[1], marker='d', ls='', c='r', label='wall_close')
    ax.plot(wall_s_close[0], wall_s_close[1], marker='o', ls='', c='r', label='wall_s_close')
    ax.set_aspect('equal')
    ax.legend()
    plt.tight_layout()
    plt.show()
    pass