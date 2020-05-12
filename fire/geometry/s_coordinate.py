#!/usr/bin/env python

"""


Created: 
"""

import logging, time

import numpy as np
import pandas as pd
from fire.geometry.geometry import logger
from fire.misc.utils import make_iterable
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d, griddata

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

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

    Returns: (closest_coords, closest_dist, closest_index) - (R, Z) coordinate of points on bounding surface closest to
             specified points, together with the distance to the bounding surface and the index of the point

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
    if np.array(r).ndim > 1:
        raise ValueError(f'Path "s" coordinate can only be calculated along a set of 1D path data')
    dr, dz = np.diff(r), np.diff(z)
    points = np.array([dr, dz]).T
    ds = np.linalg.norm(points, axis=1)
    ds = np.concatenate([[0], ds])
    s = np.cumsum(ds)
    return s



def get_s_coord_global_r(x_im, y_im, z_im=None):
    """Crude fall back method if nothing else is available"""
    s_global = np.hypot(x_im, y_im)
    logger.warning(f'In absence of machine specific plugin, using crude get_s_coord_global_r() to calc s_global coord')
    return s_global


def get_s_coord_path_ds(x_im, y_im, z_im=None):
    """Crude fall back method if nothing else is available"""
    ds = np.vstack((np.diff(x_im), np.diff(y_im), np.diff(z_im)))
    s_path = np.linalg.norm(ds)
    logger.warning(f'In absence of machine specific plugin, using get_s_coord_path_ds() to calc s_path coord')
    return s_path

if __name__ == '__main__':
    pass