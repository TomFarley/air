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

import numpy as np

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

# Use same plugin funcs for machine sector and s_path coordinate as for MAST-U
from fire.plugins.machine_plugins.tokamak_utils import (get_machine_sector, get_s_coord_path, get_tile_louvre_label,
    get_tile_louvre_index, plot_vessel_outline, plot_vessel_top_down, format_coord)

logger = logging.getLogger(__name__)


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


def get_wall_rz_coords(ds=None, false_rz_surface_boxes=None, **kwargs):
    r, z = wall_rz_coords['R'], wall_rz_coords['Z']
    if ds is not None:
        from fire.geometry.s_coordinate import interpolate_rz_coords
        r, z = interpolate_rz_coords(r, z, ds=ds, false_surface_boxes=false_rz_surface_boxes)
    return r, z

