# -*- coding: future_fstrings -*-
"""MAST specific plugin fuctions for FIRE.

"""
import logging
from collections import Sequence, Iterable

import numpy as np
from fire.geometry import cartesian_to_toroidal
from fire.utils import make_iterable

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Required: Name of plugin module (typically name of machine == name of file), needed to be located as a plugin module
machine_plugin_name = 'mast_u'
# Recommended
machine_name = 'MAST-U'  # Will be cast to lower case (and '-' -> '_') in FIRE
plugin_info = {'description': 'This plugin supplies functions for MAST-U specific operations/information'}  # exta info
# Optional
# Parameters used to label coordinates
location_labels = ['sector', 's_coord_global', 's_coord_path']
n_sectors = 12  # Used internally
# Machine specific
None

# Use same plugin funcs for machine sector and s_path coordinate as for MAST
from fire.interfaces.machine_plugins.mast import get_machine_sector
from fire.interfaces.machine_plugins.mast import get_s_coord_path

def get_s_coord_global(x_im, y_im, z_im, **kwargs):
    """Return MAST tile s coordinates for all pixels in image

    Args:
        x_im        : x coordinates of each pixel in image
        y_im        : y coordinates of each pixel in image
        z_im        : z coordinates of each pixel in image
        **kwargs    : Not used

    Returns: MAST tile 's' coordinate for each pixel in the image equal to offset major radius

    """
    logger.warning(f'Not implemented: Using incorrect global s coordinate values for MAST-U')
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