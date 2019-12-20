"""MAST specific plugin fuctions for FIRE.

"""
from collections import Sequence, Iterable

import numpy as np
from fire.geometry import cartesian_to_toroidal
from fire.utils import make_iterable

machine_plugin_name = 'mast'  # Required: Name of plugin module (typically name of machine == name of file)
machine_name = 'MAST'
plugin_info = {'description': 'This plugin supplies functions for MAST specific operations/information'}

def get_machine_sector(x, y, z=None):
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
    n_sectors = 12
    sector_angle = 360/n_sectors
    sector = (3 - np.floor(phi / sector_angle)) % n_sectors
    # No sector zero
    sector = np.where((sector == 0), 12, sector)
    if scalar:
        sector = sector[0]
    return sector

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

def get_s_coord_path(x_path, y_path, z_path):
    s_path = get_s_coord_global(x_path, y_path, z_path)
    return s_path

def format_coord(coord):
    """
    MAST coordinate formatter, includes sector number.

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