#!/usr/bin/env python

"""Generic plugin tools to be used or adapted by various other machine plugins


Created: Tom Farley, 13-01-2022
"""

import logging
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from fire.geometry.geometry import cartesian_to_toroidal
from fire.misc.utils import make_iterable
# from fire.plugins.machine_plugins.mast import n_sectors, first_sector_start_angle, sectors_clockwise, \
#     n_louvres_per_sector, get_wall_rz_coords, s_start_coord_default, surface_radii, logger

logger = logging.getLogger(__name__)

# Used internally in funcs below
n_sectors = 12
first_sector_start_angle = 90.0
sectors_clockwise = True
n_louvres_per_sector = 2

def get_machine_sector(x, y, z=None, n_sectors=n_sectors, first_sector_start_angle=first_sector_start_angle,
                       clockwise=sectors_clockwise, **kwargs):
    """Return sector number for supplied cartesian coordinates

    Args:
        x: x cartesian coordinate(s)
        y: y cartesian coordinate(s)
        z: (Optional) z cartesian coordinate(s)
        n_sectors: Number of equally sized sectors machine is divided into
        first_sector_start_angle: Angle sector 1 begins at (eg 90 deg on MAST)
        clockwise: Whether sectors are ordered clockwise (as opposed to phi which increases anticlockwise)

    Returns: Sector number(s) for supplied coordinates

    """
    if isinstance(x, Iterable):
        scalar = False
    else:
        # Convert to Iterable for masking
        scalar = True
        x, y = make_iterable(x, ndarray=True), make_iterable(y, ndarray=True)

    r, phi, theta = cartesian_to_toroidal(x, y, angles_in_deg=True, angles_positive=True)

    # MAST has 12 sectors going clockwise from North
    # Phi coordinate goes anticlockwise starting at phi=0 along x axis (due East)
    sector_width_angle = 360/n_sectors

    # sector = (3 - np.floor(phi / sector_width_angle)) % n_sectors
    sector = (first_sector_start_angle/sector_width_angle) + ((-1)**clockwise) * np.floor(phi / sector_width_angle)

    with warnings.catch_warnings():  # Suppress "invalid value encountered in remainder"
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sector = sector % n_sectors

    sector = sector.astype(int)
    # No sector zero
    sector = np.where((sector == 0), n_sectors, sector)
    # Set areas without coordinate data to -1
    sector = np.where(np.isnan(x), -1, sector)
    if scalar:
        sector = sector[0]
    return sector


def get_s_coord_path(x_path, y_path, z_path, **kwargs):
    from fire.geometry.s_coordinate import calc_local_s_along_path
    r, z, phi = cartesian_to_toroidal(x_path, y_path, z_path)
    s_path = calc_local_s_along_path(r, z)
    # s_path = get_s_coord_global(x_path, y_path, z_path)
    return s_path


def get_tile_louvre_label(x, y, z=None, n_sectors=n_sectors, n_louvres_per_sector=n_louvres_per_sector, **kwargs):
    """Each sector of MAST contains four sets of radial tiles termed louvres, labeled A-C in each sector?

    Args:
        x: x cartesian coordinate(s)
        y: y cartesian coordinate(s)
        z: (Optional) z cartesian coordinate(s)
        n_sectors: Number of sectors the machine is divided into
        n_louvres_per_sector: Number of equally sized louvres each sector is divided into

    Returns: Louvre label for each pixel in format "{sector_number}{lourvre_letter}" e.g. 3B or 11A

    """
    sector = get_machine_sector(x, y, z, n_sectors=n_sectors, n_louvres_per_sector=n_louvres_per_sector)
    louvre_index = get_tile_louvre_index(x, y, z, n_louvres_per_sector=n_louvres_per_sector)
    louvre_index_nans = np.isnan(louvre_index)
    louvre_index = np.where(louvre_index_nans, -1, louvre_index).astype(int)

    to_letter = np.vectorize(chr)
    louvre_letter = to_letter(65 + louvre_index)
    add_str_arrays = np.vectorize(str.__add__)
    louvre_label = add_str_arrays(sector.astype(int).astype(str),louvre_letter)
    louvre_label[louvre_index_nans] = ''
    return louvre_label


def get_tile_louvre_index(x, y, z=None, n_sectors=n_sectors, n_louvres_per_sector=n_louvres_per_sector, **kwargs):
    r, phi, theta = cartesian_to_toroidal(x, y, angles_in_deg=True)
    phi = np.where((phi < 0), phi + 360, phi)

    sector_width_angle = 360 / n_sectors
    louvre_width_angle = sector_width_angle / n_louvres_per_sector
    intra_sector_angle = phi % sector_width_angle
    louvre_index = intra_sector_angle // louvre_width_angle
    # Don't label points outside the divertor as belonging to louvres
    if z is not None:
        z_mask = np.abs(z) < 1.824
        louvre_index[z_mask] = np.nan
    return louvre_index


def format_coord(coord, **kwargs):
    """
    MAST coordinate formatter, includes sector number.

    Args:
        coord: Array of (x, y, z) coordinates describing a point in the machine

    Returns: String describing the position of the point within the machine

    """
    x, y, z = coord[0], coord[1], coord[2]
    r, phi = cartesian_to_toroidal(x, y)

    sector = get_machine_sector(phi)
    louvre_label = get_tile_louvre_label(x, y, z)

    formatted_coord = 'X,Y,Z: ( {:.3f} m , {:.3f} m , {:.3f} m )'.format(x, y, z)
    formatted_coord = formatted_coord + u'\nR,Z,\u03d5: ( {:.3f} m , {:.3f}m , {:.1f}\xb0 )'.format(r, z, phi)
    formatted_coord = formatted_coord + '\n Sector {:.0f}'.format(sector)
    formatted_coord = formatted_coord + '\n Louvre {}'.format(louvre_label)

    return formatted_coord


def plot_vessel_outline(ax=None, top=True, bottom=True, aspect='equal', ax_labels=True,
                              axes_off=False, show=True, **kwargs):
    from fire.plotting.plot_tools import plot_vessel_outline as plot_vessel_outline_rz

    r, z = get_wall_rz_coords()
    fig, ax = plot_vessel_outline_rz(r, z, ax=ax, top=top, bottom=bottom, aspect=aspect, ax_labels=ax_labels,
                                axes_off=axes_off, show=show, s_start_coord=s_start_coord_default)
    return fig, ax


def plot_vessel_top_down(ax=None, keys_plot=('R_T1', 'R_T2', 'R_T3', 'R_T3_top'),
                         keys_plot_strong=('R_T1', 'R_T3_top'),
            axes_off=False, phi_labels=True):

    from fire.plotting.plot_tools import plot_vessel_top_down

    fig, ax = plot_vessel_top_down(surface_radii, keys_plot, ax=ax, axes_off=axes_off, phi_labels=phi_labels,
                                   keys_plot_strong=keys_plot_strong)
    logger.warning('Plotting top down view of MAST with placeholder tile radii')

    return fig, ax


if __name__ == '__main__':
    pass