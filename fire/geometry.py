#!/usr/bin/env python

"""


Created: 
"""

import logging
from typing import Union, Iterable, Tuple, Optional
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def identify_visible_structures(r, phi, z, surface_coords, phi_in_deg=True):
    # Create mask with values corresponding to id of structure visible in each pixel
    if not phi_in_deg:
        phi = np.rad2deg(phi)
    bg_value = np.nan
    structure_ids = np.full_like(r, bg_value)
    visible_structures = {}
    for surface_name, row in surface_coords.iterrows():
        surface_id = row['id']
        periodicity = row['toroidal_periodicity']
        r_range = np.sort([row['R1'], row['R2']])
        phi_range = np.sort([row['phi1'], row['phi2']])
        z_range = np.sort([row['z1'], row['z2']])
        for i in np.arange(periodicity):
            # Copy structure about machine with given periodicity
            # TODO: Keep track of toroidal id 'i'?
            phi_range_i = (phi_range + 360*i/periodicity) % 360
            if phi_range_i.sum() == 0:
                # Account for 360%0=0
                phi_range_i[1] = 360
            mask = (((r >= r_range[0]) & (r <= r_range[1])) & ((phi >= phi_range_i[0]) & (phi <= phi_range_i[1])) &
                    ((z >= z_range[0]) & (z <= z_range[1])))
            # Reflect structure in z=0 plane for up down symetric components
            if row['mirror_z']:
                mask += (((r >= r_range[0]) & (r <= r_range[1])) & ((phi >= phi_range_i[0]) & (phi <= phi_range_i[1])) &
                        ((z >= -z_range[1]) & (z <= -z_range[0])))
        if any(~np.isnan(structure_ids[mask])):
            raise ValueError(f'Surface coordinates overlap. Previously assigned pixels reassigned to id: '
                             f'"{surface_id}", structure: "{surface_name}"')
        structure_ids[mask] = surface_id
        if np.sum(mask) > 0:
            visible_structures[surface_id] = surface_name
    if len(visible_structures) == 0:
        raise ValueError(f'No surfaces identified in camera view')
    return structure_ids, visible_structures


def load_material_properties(visible_surfaces):
    tile_properties = []
    for tile in set(analysis_path_tiles):
        tile_data = read_csv()
        tile_properties.append(tile_data)
    tile_properties = xr.DataArray(coords=analysis_path.coords)
    raise NotImplementedError

def setgment_path_by_material():
    raise NotImplementedError
    # TODO: Read path_fn_tile_coords
    tile_names = np.full_like(r, '', dtype=object)
    tile_coords = read_csv(path_fn=path_fn_tile_coords, index_col='tile_name')

    no_tile_info_mask = tile_name == ''
    if any(no_tile_info_mask):
        tile_names[no_tile_info_mask] = np.nan
        if raise_on_no_tile_info:
            raise ValueError(
                f'Analysis path contains {np.sum(no_tile_info_mask)}/{len(r)} points without tile info:\n'
                f'r={r[no_tile_info_mask]}\nz={z[no_tile_info_mask]}')
    return tile_names


if __name__ == '__main__':
    pass