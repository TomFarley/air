# -*- coding: future_fstrings -*-
"""MAST specific plugin fuctions for FIRE.

"""
import logging

import numpy as np
from fire.plotting.plot_tools import format_poloidal_plane_ax
from fire.geometry.s_coordinate import interpolate_rz_coords, separate_rz_points_top_bottom, calc_s_coord_lookup_table, \
    get_nearest_s_coordinates, get_nearest_rz_coordinates, get_nearest_boundary_coordinates

from fire.geometry.geometry import cartesian_to_toroidal
from fire.misc.utils import make_iterable

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

module_default = object()

# Required: Name of plugin module (typically name of machine == name of file), needed to be located as a plugin module
machine_plugin_name = 'mast_u'

# Recommended
machine_name = 'MAST-U'  # Will be cast to lower case (and '-' -> '_') in FIRE
plugin_info = {'description': 'This plugin supplies functions for MAST-U specific operations/information'}  # exta info
location_labels = ['sector', 's_global']  # Parameters used to label coordinates

# Optional/other
n_sectors = 12  # Used internally in funcs below
# Machine specific
None

# Boxes to pass to fire.s_coordinate.remove_false_rz_surfaces
false_rz_surface_boxes_default = [((1.512, 1.6), (-0.81789, 0.81789)),
                                  ]
s_start_coord_default = (0.260841, 0)

# Use same plugin funcs for machine sector and s_path coordinate as for MAST
from fire.interfaces.machine_plugins.mast import get_machine_sector


def get_uda_mastu_wall_coords(no_cal=True, signal="/limiter/efit", shot=50000):
    """Return (R, Z) coordinates of points defining wall outline of MAST-U tile surfaces

    This is normally safe to call with default arguments.

    Args:
        no_cal: Whether to return idealised CAD coordinates without spatial calibration corrections
        signal: UDA signal for wall coords
        shot: Shot number to get wall definition for

    Returns: Tuple of (R, Z) coordinate arrays

    """
    import pyuda
    client=pyuda.Client()
    wall_data = client.geometry(signal, shot, no_cal=no_cal)
    r = wall_data.data.R
    z = wall_data.data.Z
    return r, z

def get_tile_edge_coords_mastu(no_cal=True, signal="/limiter/efit", shot=50000):
    """Return (R,Z) coords of main tile boundaries

    Args:
        no_cal: Whether to use idealised CAD coordinates without spatial calibration corrections
        signal: UDA signal for wall coords
        shot: Shot number to get wall definition for

    Returns: Tuple of (R, Z) coordinate arrays

    """
    min_tile_size = 0.10
    r0, z0 = get_uda_mastu_wall_coords(signal=signal, no_cal=no_cal, shot=shot)
    diff = np.linalg.norm([np.diff(r0), np.diff(z0)], axis=0)
    mask = (diff > min_tile_size) | (np.roll(diff, -1) > min_tile_size)
    mask = np.concatenate([[np.True_], mask])
    r_tiles, z_tiles = r0[mask], z0[mask]
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
    r0, z0 = get_uda_mastu_wall_coords(no_cal=no_cal, signal=signal, shot=shot)
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

def get_nearest_s_coordinates_mastu(r, z, tol=5e-3, ds=1e-3, no_cal=True, signal="/limiter/efit", shot=50000):
    """Return closest tile surface 's' coordinates for supplied (R, Z) coordinates

    Args:
        r: Array of radial R coordinates
        z: Array of vertical Z coordinates
        tol: Tolerance distance for points from wall - return nans if further away than tolerance
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


def plot_tile_edges_mastu(ax=None, show=True):
    if ax is None:
        fig, ax = create_poloidal_cross_section_figure(1, 1)
    else:
        fig = ax.figure

    r0, z0 = get_tile_edge_coords_mastu(no_cal=True, shot=50000)

    ax.plot(r0, z0, marker='x', ls='')

    if show:
        plt.show()
    return fig, ax

def plot_vessel_outline_mastu(ax=None, top=True, bottom=True, show=True):
    if ax is None:
        fig, ax = create_poloidal_cross_section_figure(1, 1)
    else:
        fig = ax.figure

    r0, z0 = get_uda_mastu_wall_coords(no_cal=True, shot=50000)
    (r_bottom, z_bottom), (r_top, z_top) = separate_rz_points_top_bottom(r0, z0)
    # ax.plot(r0, z0, marker='x', ls='-')
    if bottom:
        ax.plot(r_bottom, z_bottom, marker='', ls='-')
    if top:
        ax.plot(r_top, z_top, marker='', ls='-')
    if show:
        plt.tight_layout()
        ax.set_aspect('equal')
        plt.show()
    return fig, ax

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

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # fig, ax = plot_vessel_outline_mastu(show=True, top=False)
    # plot_tile_edges(ax=ax, show=True)
    ds = 1e-4
    # ds = 1e-2
    r0, z0 = get_uda_mastu_wall_coords(no_cal=True, shot=50000)
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