"""MAST specific plugin functions for FIRE.

Functions and top level variables in this file with appropriate names will be imported and used by FIRE to provide
functionality and information specific to this machine.

The most important role of the machine plugin is to provide a means of looking up the tile surface 's' coordinate
against which heat flux profiles etc are defined. Additional machine specific information and tools include
identifying sector numbers and other location based labels and plotting methods for visualising this information.

Author: Tom Farley (tom.farley@ukaea.uk)
Created: 08-2019
"""
import logging
import warnings
from collections import Iterable

import numpy as np
from fire.geometry.geometry import cartesian_to_toroidal
from fire.misc.utils import make_iterable

# Required: Name of plugin module (typically name of machine == name of file), needed to be located as a plugin module
machine_plugin_name = 'mast'
# Recommended
machine_name = 'MAST'  # Will be cast to lower case in FIRE
plugin_info = {'description': 'This plugin supplies functions for MAST specific operations/information'}  # exta info
# Optional
# Parameters used to label coordinates across the whole image
location_labels_im = ['sector', 's_global', 'louvre_index', 'louvre_label']
# Parameters defined along a specific analysis path through an image
location_labels_path = ['s_path']
n_sectors = 12  # Used internally
first_sector_start_angle = 90.0
sectors_clockwise = True
# Machine specific
n_louvres_per_sector = 4

# Origin for s coordinates/divider between top and bottom of machine
s_start_coord_default = (0.260841, 0)
# Radii of edges of different structures for top down view plot etc
surface_radii = {'R_wall': 2.0,
                'R_HL04': 2.1333,
                'R_T1': 0.333,  # TODO: Replace with correct tile raddi values
                'R_T2': 0.7,
                'R_T3': 1.1,
                'R_T3_top': 1.5,
                'n_sectors': 12,
                }

# Wall coordinates taken from MAST gfile
wall_rz_coords = {'R': np.array([1.9, 1.555104, 1.555104, 1.407931, 1.407931, 1.039931, 1.039931,
                 1.9, 1.9, 0.564931, 0.564931, 0.7835, 0.7835, 0.58259,
                 0.4165, 0.28, 0.28, 0.195244, 0.195244, 0.28, 0.28,
                 0.4165, 0.58259, 0.7835, 0.7835, 0.564931, 0.564931, 1.9,
                 1.9, 1.039931, 1.039931, 1.407931, 1.407931, 1.555104, 1.555104,
                 1.9, 1.9]),
                'Z': np.array([0.405, 0.405, 0.8225, 0.8225, 1.033, 1.033,
                 1.195, 1.195, 1.825, 1.825, 1.728082, 1.728082,
                 1.715582, 1.547, 1.547, 1.6835, 1.229089, 1.0835,
                 -1.0835, -1.229089, -1.6835, -1.547, -1.547, -1.715582,
                 -1.728082, -1.728082, -1.825, -1.825, -1.195, -1.195,
                 -1.033, -1.033, -0.8225, -0.8225, -0.405, -0.405,
                 0.405])}

logger = logging.getLogger(__name__)

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

def get_s_coord_global(x_im, y_im, z_im, **kwargs):
    """Return MAST tile s coordinates for all pixels in image

    Args:
        x_im        : x coordinates of each pixel in image
        y_im        : y coordinates of each pixel in image
        z_im        : z coordinates of each pixel in image
        **kwargs    : Not used

    Returns: MAST tile 's' coordinate for each pixel in the image equal to offset major radius

    """
    r_t1_start = 0.757
    r_t3_end = 1.658
    z_tiles = 1.824
    r_im = np.hypot(x_im, y_im)
    s_global = np.array(r_im - r_t1_start)
    # Set regions away from tiles to nan
    s_global[r_im < r_t1_start] = np.nan
    s_global[r_im > r_t3_end] = np.nan
    s_global[np.abs(z_im) < z_tiles] = np.nan
    return s_global

def get_s_coord_path(x_path, y_path, z_path, **kwargs):
    from fire.geometry.s_coordinate import calc_local_s_along_path
    r, z, phi = cartesian_to_toroidal(x_path, y_path, z_path)
    s_path = calc_local_s_along_path(r, z)
    # s_path = get_s_coord_global(x_path, y_path, z_path)
    return s_path

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

def get_wall_rz_coords(ds=None, false_rz_surface_boxes=None, **kwargs):
    r, z = wall_rz_coords['R'], wall_rz_coords['Z']
    if ds is not None:
        from fire.geometry.s_coordinate import interpolate_rz_coords
        r, z = interpolate_rz_coords(r, z, ds=ds, false_surface_boxes=false_rz_surface_boxes)
    return r, z

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
