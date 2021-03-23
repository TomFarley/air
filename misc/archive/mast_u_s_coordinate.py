#!/usr/bin/env python

"""Functions for working with MAST-U tile surface s coordinates


Tom Farley, April 2020
"""

import logging, time
from typing import Union, Iterable, Sequence, Tuple, Optional, Any, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d, griddata

from fire.plotting.plot_tools import create_poloidal_cross_section_figure

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# Boxes to pass to fire.s_coordinate.remove_false_rz_surfaces
false_rz_surface_boxes_default = [((1.512, 1.6), (-0.81789, 0.81789)),
                                  ]
s_start_coord_default = (0.260841, 0)

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
    # s = xr.merge([s_bottom, s_top])  # , combine_attrs='no_conflicts')
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
    z_mask = z <= 0
    s = np.full_like(r, np.nan, dtype=float)
    position = np.full_like(r, np.nan, dtype=float)
    table_key = {-1: 's_bottom', 1: 's_top'}
    for mask, key, pos in zip([z_mask, ~z_mask], ['s_bottom', 's_top'], [-1, 1]):
        lookup_table = s_lookup[key]
        if np.any(mask):
            r_masked, z_masked = r[mask], z[mask]
            r_wall, z_wall, s_wall = lookup_table['R'], lookup_table['Z'], lookup_table['s']
            s[mask] = get_nearest_s_coordinates(r_masked, z_masked, r_wall, z_wall, s_wall, tol=tol)
            position[mask] = pos
    return s, (position, table_key)


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

def interpolate_rz_coords(r, z, ds=1e-4, tol_abs=5e-5, tol_interger_spacing=5e-2, false_surface_boxes=None):
    """Return interpolated arrays of R, Z coordinates with points separated from each other by the spacing ds

    Where input points are separated by a non-integer ds spacing, both original end points are returned if the last of
    points separated by ds would deviate from the original end point by bore that the supplied tolerances

    Args:
        r: Radial R coordinates
        z: Vertical Z coordinates
        ds: Desired spacing of output coordinates
        tol_abs: Maximum absolute loss in precision to input points to avoid smoothing edges in supplied points
        tol_interger_spacing: Fraction of ds at which to limit loss of accuracy on orignal points(default 5%)
        false_surface_boxes: List of box coordinates where interpolated points should be removed (ie no points exist)
                             in form: [([R1,R2], [Z1, Z2]),  ... , ([R1, R2], [Z1, Z2])]

    Returns: Interpolated arrays of R, Z coordinates with points separated from each other by the spacing ds

    """
    r_line_sections = []
    z_line_sections = []
    for r1, z1, r2, z2 in zip(r, z, r[1:], z[1:]):
        # Linear interpolation between each pair of points
        dr = r2-r1
        dz = z2-z1
        ds_original = np.linalg.norm([dr, dz])
        n_points_in_section = ds_original / ds
        n_points_in_section_int = int(n_points_in_section)
        if dr == 0:
            z_interpolated = np.arange(z1, z2, ds*np.sign(dz))
            if z_interpolated[-1] != z2:
                z_interpolated = np.concatenate([z_interpolated, [z2]])
            r_interpolated = np.full_like(z_interpolated, r1)
        else:
            if np.isclose(n_points_in_section, n_points_in_section_int, atol=tol_abs, rtol=tol_interger_spacing):
                # Integer number of ds between points (to within 5% of ds)
                r_interpolated = np.linspace(r1, r2, n_points_in_section_int)
            else:
                # R coordinate at end of integer number of ds
                r2_int = r1 + (dr) * (int(n_points_in_section)/n_points_in_section)
                r_interpolated = np.linspace(r1, r2_int, int(n_points_in_section))
                # Include end of line so have correct corners
                r_interpolated = np.concatenate([r_interpolated, [r2]])
            f = interp1d([r1, r2], [z1, z2], assume_sorted=False, fill_value="extrapolate")
            z_interpolated = f(r_interpolated)
        r_line_sections.append(r_interpolated)
        z_line_sections.append(z_interpolated)
        # debug = True
        debug = False
        if debug:
            dr_interpolated, dz_interpolated = np.diff(r_interpolated), np.diff(z_interpolated)
            points = np.array([dr_interpolated, dz_interpolated]).T
            ds_interpolated = np.linalg.norm(points, axis=1)
            s_interpolated = np.cumsum(ds_interpolated)
            print(points)
            print(ds_interpolated)
            print(s_interpolated)
            fig, ax = plt.subplots()
            ax.plot(r_interpolated, z_interpolated, marker='x', ls='')
            ax.plot(r, z, marker='x', ls='')
            ax.plot([r1, r2], [z1, z2], marker='x', ls='')
            plt.tight_layout()
            plt.show()
            pass
    r_interpolated = np.concatenate(r_line_sections)
    z_interpolated = np.concatenate(z_line_sections)
    if false_surface_boxes is not None:
        r_interpolated, z_interpolated = remove_false_rz_surfaces(r_interpolated, z_interpolated,
                                                                  remove_boxes=false_surface_boxes)

    return r_interpolated, z_interpolated


def remove_false_rz_surfaces(r, z, remove_boxes):
    """Filter out (R, Z) coordinates within supplied R, Z boxes

    Args:
        r: R coordinates
        z: Z coordinates
        remove_boxes: List of box coordinates in form [([R1, R2], [Z1, Z2]),  ... , ([R1, R2], [Z1, Z2])]

    Returns: Filtered R, Z arrays

    """
    for (r1, r2), (z1, z2) in remove_boxes:
        mask_in_box = ((r > r1) & (r < r2) & (z > z1) & (z < z2))
        mask_keep = ~mask_in_box
        r = r[mask_keep]
        z = z[mask_keep]
    return r, z


def separate_rz_points_top_bottom(r, z, top_coord_start, bottom_coord_start, prepend_start_coord=True):
    """Split arrays of R, Z surface coordinates into separate sets for the top and bottom of the machine.

    The returned arrays are ordered so that they start at the point closest to the supplied start coordinate and the
    start coordinate is prepended to the returned arrays if prepend_start_coord=True.
    This is useful for separating tile wall coordinates used to calculate tile surface 's' coordinates.

    Args:
        r: Array of R coordinates to be top-bottom separated
        z: Array of Z coordinates to be top-bottom separated
        top_coord_start: (R, Z) coordinate that chain of 'top' coordinates should originate from
        bottom_coord_start: (R, Z) coordinate that chain of 'bottom' coordinates should originate from
        prepend_start_coord: Whether to prepend start coordinate to reuturned arrays

    Returns: R, Z coordinates separated into (r_bottom, z_bottom), (r_top, z_top)

    """
    mask_bottom = z <= 0
    r_bottom = r[mask_bottom]
    z_bottom = z[mask_bottom]
    r_top = r[~mask_bottom]
    z_top = z[~mask_bottom]

    # Make sure returned arrays are ordered starting closest to the specified coordinate (ie clockwise/anticlockwise)
    if bottom_coord_start is not None:
        r_bottom, z_bottom = order_arrays_starting_close_to_point(r_bottom, z_bottom, *bottom_coord_start,
                                                               prepend_start_coord=prepend_start_coord)
    if top_coord_start is not None:
        r_top, z_top = order_arrays_starting_close_to_point(r_top, z_top, *top_coord_start,
                                                               prepend_start_coord=prepend_start_coord)

    return (r_bottom, z_bottom), (r_top, z_top)


def order_arrays_starting_close_to_point(r, z, r_start, z_start, prepend_start_coord=True):
    distance_from_start = np.linalg.norm([r - r_start, z - z_start], axis=0)
    i_closest = np.argmin(distance_from_start)
    if distance_from_start[i_closest+1] > distance_from_start[i_closest-1]:
        # Switch clockwise/anticlockwise
        r = r[::-1]
        z = z[::-1]
        i_closest = len(r) - i_closest - 1
    # Start array with point closest to start coordinate
    r = np.roll(r, -i_closest)
    z = np.roll(z, -i_closest)

    if prepend_start_coord:
        if (r[0], z[0]) != (r_start, z_start):
            r = np.concatenate([[r_start], r])
            z = np.concatenate([[z_start], z])
    return r, z


def calc_s_coord_lookup_table(r, z):
    s = calc_local_s_along_path(r, z)
    s = pd.DataFrame.from_dict({'R': r, 'Z': z, 's': s})

    # s = pd.Series(s, index=pd.MultiIndex.from_arrays((r, z), names=['R', 'Z']), name='s')
    # s = s.loc[~s.index.duplicated(keep='first')]

    # s = xr.DataArray(s, dims=['i'], coords={'i': np.arange(len(s)), 'R': ('i', r), 'Z': ('i', z),
    #                                         # 'RZ': ('i', np.array([r, z]).T)
    #                                         },
    #                  name='s',
    #                  attrs={'symbol': '$s$', 'units': 'm', 'longname': 'Divertor s coordinate'})
    # s = s.stack(RZ=('R', 'Z'))
    return s


def get_nearest_s_coordinates(r, z, r_wall, z_wall, s_wall, tol=None):
    r, z = make_iterable(r, ndarray=True), make_iterable(z, ndarray=True)
    s_closest = griddata((r_wall, z_wall), s_wall, (r, z), method='nearest')
    if tol is not None:
        closest_coords, closest_dist, closest_index = get_nearest_boundary_coordinates(r, z, r_wall, z_wall)
        mask = closest_dist > tol
        s_closest[mask] = np.nan
    return s_closest


def get_nearest_rz_coordinates(s, r_wall, z_wall, s_wall):
    """Return (R, Z) wall coordinates of supplied tile surface 's' coordinates"""
    s = make_iterable(s, ndarray=True)
    f_r = interp1d(s_wall, r_wall)
    f_z = interp1d(s_wall, z_wall)
    r = f_r(s)
    z = f_z(s)
    return (r, z)

def get_nearest_boundary_coordinates(r, z, r_boundary, z_boundary):
    """Return boundary coordinate closest to supplied point.

    Args:
        r: R coordinate of interest
        z: Z coordinate of interest
        r_boundary: R coordinates of bounding surface/wall to look up closest point on
        z_boundary: Z coordinates of bounding surface/wall to look up closest point on

    Returns: (R, Z) coordinate of point on bounding surface closest to specified point

    """
    from scipy.spatial import distance
    r, z = make_iterable(r, ndarray=True), make_iterable(z, ndarray=True)
    points = np.array([r, z]).T
    boundaries = np.array([r_boundary, z_boundary]).T
    t0 = time.time()
    distances = distance.cdist(points, boundaries)
    t1 = time.time()
    logger.debug(f'Calculated {distances.size} distances from {len(r)} points to {len(r_boundary)} boundary points in '
                 f'{t1-t0:0.3f}s')
    closest_index = distances.argmin(axis=1)
    closest_dist = distances[np.arange(len(r)), closest_index]
    closest_coords = boundaries[closest_index]
    return closest_coords, closest_dist, closest_index


def calc_local_s_along_path(r, z):
    """Return path length along path specified by (R, Z) coords"""
    dr, dz = np.diff(r), np.diff(z)
    points = np.array([dr, dz]).T
    ds = np.linalg.norm(points, axis=1)
    ds = np.concatenate([[0], ds])
    s = np.cumsum(ds)
    return s

def make_iterable(obj: Any, ndarray: bool = False,
                  cast_to: Optional[type] = None,
                  cast_dict: Optional = None,
                  # cast_dict: Optional[dict[type,type]]=None,
                  nest_types: Optional = None) -> Iterable:
    # nest_types: Optional[Sequence[type]]=None) -> Iterable:
    """Return itterable, wrapping scalars and strings when requried.

    If object is a scalar nest it in a list so it can be iterated over.
    If ndarray is True, the object will be returned as an array (note avoids scalar ndarrays).

    Args:
        obj         : Object to ensure is iterable
        ndarray     : Return as a non-scalar np.ndarray
        cast_to     : Output will be cast to this type
        cast_dict   : dict linking input types to the types they should be cast to
        nest_types  : Sequence of types that should still be nested (eg dict)

    Returns:

    """
    if not hasattr(obj, '__iter__') or isinstance(obj, str):
        obj = [obj]
    if (nest_types is not None) and isinstance(obj, nest_types):
        obj = [obj]
    if (cast_dict is not None) and (type(obj) in cast_dict):
        obj = cast_dict[type(obj)](obj)
    if ndarray:
        obj = np.array(obj)
    if isinstance(cast_to, type):
        if cast_to == np.ndarray:
            obj = np.array(obj)
        else:
            obj = cast_to(obj)  # cast to new type eg list
    return obj

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
    s_close, (pos, table_key) = get_nearest_s_coordinates_mastu(*point)
    s_close_table = s_tables[table_key[pos[0]]]
    wall_s_close = get_nearest_rz_coordinates(s_close, s_close_table['R'].values, s_close_table['Z'].values,
                                              s_close_table['s'].values)
    wall_close, closest_dist, closest_index = get_nearest_boundary_coordinates(*point, r_boundary=r, z_boundary=z)
    fig, ax = plt.subplots(1, 1)
    ax.plot(r, z, marker='x', ls='')
    ax.plot(r0, z0, marker='x', ls='')
    # ax.plot(r0[:10], z0[:10], marker='x', ls='')
    ax.plot(point[0], point[1], marker='x', ls='', c='r', label='point')
    ax.plot(wall_close[0, 0], wall_close[0, 1], marker='d', ls='', c='r', label='wall_close')
    ax.plot(wall_s_close[0], wall_s_close[1], marker='o', ls='', c='r', label='wall_s_close')
    ax.set_aspect('equal')
    ax.legend()
    plt.tight_layout()
    plt.show()
    pass