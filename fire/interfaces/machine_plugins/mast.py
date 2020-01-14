"""MAST specific plugin fuctions for FIRE.

"""
from collections import Sequence, Iterable

import numpy as np
from fire.geometry import cartesian_to_toroidal
from fire.utils import make_iterable

# Required: Name of plugin module (typically name of machine == name of file), needed to be located as a plugin module
machine_plugin_name = 'mast'
# Recommended
machine_name = 'MAST'  # Will be cast to lower case in FIRE
plugin_info = {'description': 'This plugin supplies functions for MAST specific operations/information'}  # exta info
# Optional
# Parameters used to label coordinates
location_labels = ['sector', 's_coord_global', 's_coord_path', 'louvre_index', 'louvre_label']
n_sectors = 12  # Used internally
# Machine specific
n_louvres_per_sector = 4

def get_machine_sector(x, y, z=None, **kwargs):
    """Return MAST sector number for supplied cartesian coordinates

    Args:
        x: x cartesian coordinate(s)
        y: y cartesian coordinate(s)
        z: (Optional) z cartesian coordinate(s)

    Returns: Sector number(s) for supplied coordinates

    """
    if isinstance(x, Iterable):
        scalar = False
    else:
        # Convert to Iterable for masking
        scalar = True
        x, y = make_iterable(x, ndarray=True), make_iterable(y, ndarray=True)

    r, phi = cartesian_to_toroidal(x, y, phi_in_deg=True)
    phi = np.where((phi < 0), phi+360, phi)

    # MAST has 12 sectors going clockwise from North
    # Phi coordinate goes anticlockwise starting at phi=0 along x axis (due East)
    sector_width_angle = 360/n_sectors
    sector = (3 - np.floor(phi / sector_width_angle)) % n_sectors
    # No sector zero
    sector = np.where((sector == 0), 12, sector)
    if scalar:
        sector = sector[0]
    return sector

def get_tile_louvre_label(x, y, z=None, **kwargs):
    """Each sector of MAST contains four sets of radial tiles termed louvres, labeled A-C in each sector?

    Args:
        x: x cartesian coordinate(s)
        y: y cartesian coordinate(s)
        z: (Optional) z cartesian coordinate(s)

    Returns: Louvre label for each pixel in format "{sector_number}{lourvre_letter}" e.g. 3B or 11A

    """
    sector = get_machine_sector(x, y, z)
    louvre_index = get_tile_louvre_index(x, y, z)
    louvre_index_nans = np.isnan(louvre_index)
    louvre_index = np.where(louvre_index_nans, -1, louvre_index).astype(int)
    to_letter = np.vectorize(chr)
    louvre_letter = to_letter(65 + louvre_index)
    add_str_arrays = np.vectorize(str.__add__)
    louvre_label = add_str_arrays(sector.astype(int).astype(str),louvre_letter)
    louvre_label[louvre_index_nans] = ''
    return louvre_label

def get_tile_louvre_index(x, y, z=None, **kwargs):
    r, phi = cartesian_to_toroidal(x, y, phi_in_deg=True)
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
    s_path = get_s_coord_global(x_path, y_path, z_path)
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