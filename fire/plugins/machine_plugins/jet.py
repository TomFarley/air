# -*- coding: future_fstrings -*-
"""JET specific plugin functions for FIRE.

Functions and top level variables in this file with appropriate names will be imported and used by FIRE to provide
functionality and information specific to this machine.

The most important role of the machine plugin is to provide a means of looking up the tile surface 's' coordinate
against which heat flux profiles etc are defined. Additional machine specific information and tools include
identifying sector numbers and other location based labels and plotting methods for visualising this information.

Author: Tom Farley (tom.farley@ukaea.uk)
Created: 05-05-2020
"""
import logging

import numpy as np
from fire.plotting.plot_tools import create_poloidal_cross_section_figure
from fire.geometry.s_coordinate import interpolate_rz_coords, separate_rz_points_top_bottom, get_nearest_rz_coordinates, get_nearest_boundary_coordinates

from fire.geometry.geometry import cartesian_to_toroidal

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# ===================== PLUGIN MODULE ATTRIBUTES =====================
# Required: Name of plugin module (typically name of machine == name of file), needed to be located as a plugin module
machine_plugin_name = 'jet'

# Recommended
machine_name = 'JET'  # Will be cast to lower case (and '-' -> '_') in FIRE
plugin_info = {'description': 'This plugin supplies functions for JET specific operations/information'}  # extra info
location_labels_im = ['sector', 's_global']  # Parameters used to label coordinates across the whole image

# Optional/other
n_sectors = 8  # Used internally in funcs below
# Machine specific
None

# Boxes to pass to fire.s_coordinate.remove_false_rz_surfaces
false_rz_surface_boxes_default = []
s_start_coord_default = (0, 0)

# Use same plugin funcs for machine sector and s_path coordinate as for MAST
from fire.plugins.machine_plugins.tokamak_utils import get_machine_sector as get_machine_sector_mast
from fire.plugins.machine_plugins.tokamak_utils import get_s_coord_path
from fire.plugins.machine_plugins.jet_tools.scoord import get_s_definition

def get_machine_sector(*args, **kwargs):
    return get_machine_sector_mast(*args, n_sectors=n_sectors, **kwargs)
# See bottom of file for function aliases
# ====================================================================

module_default = object()

def get_wall_rz_coords():
    # Load the S coordinate defition.
    s, sR, sZ = get_s_definition(wall)
    return sR, sZ

def get_s_coord_global(x_im, y_im, z_im, **kwargs):
    """Return JET tile s coordinates for all pixels in image.

    This 's' coordinate is considered 'global' as it is predefined for all (R, Z) surfaces as apposed to a 'local' s
    starting at 0m along a specific path.

    Args:
        x_im        : x coordinates of each pixel in image
        y_im        : y coordinates of each pixel in image
        z_im        : z coordinates of each pixel in image
        **kwargs    : Arguments passed to ...

    Returns: JET tile 's' coordinate for each pixel in the image

    """
    logger.warning(f'\n\nUsing incorrect JET divertor "s" coordinates - NOT IMPLEMENTED\n\n')

    r_im, phi_im, theta_im = cartesian_to_toroidal(x_im, y_im, z_im)
    s_im = r_im
    return s_im

def get_jet_wall_coords(shot=50000, ds=None):
    """Return (R, Z) coordinates of points defining wall outline of tile surfaces

    This is normally safe to call with default arguments.

    Args:
        shot: Shot number to get wall definition for

    Returns: Tuple of (R, Z) coordinate arrays

    """
    raise NotImplementedError
    if ds is not None:
        r, z = interpolate_rz_coords(r, z, ds=ds, false_surface_boxes=false_rz_surface_boxes_default)
    return r, z

def get_tile_edge_coords_jet(shot=50000, subset=True):
    """Return (R,Z) coords of main tile boundaries

    Args:
        shot: Shot number to get wall definition for

    Returns: Tuple of (R, Z) coordinate arrays

    """
    raise NotImplementedError

    return r_tiles, z_tiles

def get_nearest_s_coordinates_jet(r, z, tol=5e-3, ds=1e-3, shot=50000):
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
    raise NotImplementedError
    return s, (position, table_key)



def plot_vessel_outline_jet(ax=None, top=True, bottom=True, shot=50000, no_cal=False, aspect='equal', ax_labels=True,
                              axes_off=False, show=True, **kwargs):
    import matplotlib.pyplot as plt
    raise NotImplementedError
    return fig, ax


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

    pass