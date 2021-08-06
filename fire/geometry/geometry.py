# -*- coding: future_fstrings -*-
#!/usr/bin/env python

"""


Created: 
"""

import logging
from pathlib import Path
from collections import namedtuple

import numpy as np
import pandas as pd
import xarray as xr

import fire
import fire.interfaces.io_basic
import fire.interfaces.read_user_fire_config
from fire.misc import utils
from fire.misc.utils import ndarray_0d_to_scalar
from pyEquilibrium.equilibrium import equilibrium
import pyuda
client = pyuda.Client()
import time as clock
from fire.interfaces import interfaces, io_utils

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

def locate_structure_coords_file(machine='mast_u', paths=None, fns=None):
    if (paths is None) or (fns is None):
        config = fire.interfaces.read_user_fire_config.read_user_fire_config()
        paths = config['paths_input']['input_files']
        fns = config['filenames_input']['structure_coords']

    kws = dict(machine=machine, fire_path=fire.fire_paths['root'])
    path_fn = io_utils.locate_file(paths, fns=fns, path_kws=kws, fn_kws=kws)
    # print(path_fn)
    path_fn = path_fn[0] / path_fn[1]
    # print(path_fn, path_fn.suffix)
    return path_fn

def read_structure_coords(path_fn=None, machine='mast_u'):
    if path_fn is None:
        path_fn = locate_structure_coords_file(machine=machine)
    path_fn = Path(path_fn)
    if path_fn.suffix == '.json':
        structure_coords = fire.interfaces.io_basic.json_load(path_fn)
    elif path_fn.suffix == '.csv':
        structure_coords = fire.interfaces.io_basic.read_csv(path_fn, sep=', ', index_col='structure')
    return path_fn, structure_coords

def structure_coords_dict_to_df(structure_coords):
    from fire.misc.utils import is_scalar, is_number
    StructureCoordsTuple = namedtuple('StructureCoordsTuple', ('structure_coords_df', 'structure_coords_info'))
    if isinstance(structure_coords, pd.DataFrame):
        return StructureCoordsTuple(structure_coords, {})

    df = pd.DataFrame()
    for id, values in structure_coords.items():
        for col, value in values.items():
            if is_scalar(value):
                df.loc[id, col] = value
            elif isinstance(value, dict):
                for i, val in enumerate(value):
                    df.loc[id, f'{col}{i+1}'] = val
        # raise NotImplementedError
    return StructureCoordsTuple(df, structure_coords)

def identify_visible_structures(r_im, phi_im, z_im, surface_coords, phi_in_deg=True, bg_value = -1):
    # Create mask with values corresponding to id of structure visible in each pixel
    if not phi_in_deg:
        phi_im = np.rad2deg(phi_im)
    if np.min(phi_im) < 0:
        raise ValueError(f'Phi range is [{np.min(phi_im)}, {np.max(phi_im)}]. Expected [0, 360].')

    structure_ids = np.full_like(r_im, bg_value, dtype=type(bg_value))
    material_ids = np.full_like(r_im, bg_value, dtype=type(bg_value))
    visible_structures = {}
    visible_materials = {}

    # TODO: Switch to using json structure coords file
    # from fire.geometry.geometry import structure_coords_dict_to_df
    # surface_coords = structure_coords_dict_to_df(surface_coords)

    for surface_name, row in surface_coords.iterrows():
        surface_id = row['id']
        material_name = row['material']
        periodicity = row['toroidal_periodicity']
        r_range = np.sort([row['R1'], row['R2']])
        phi_range = np.sort([row['phi1'], row['phi2']])
        z_range = np.sort([row['z1'], row['z2']])

        if material_name in visible_materials.values():
            keys, values = list(visible_materials.keys()), list(visible_materials.values())
            material_id = keys[values.index(material_name)]
        else:
            material_id = len(visible_materials) + 1

        for i in np.arange(periodicity):
            # Copy structure about machine with given periodicity
            # TODO: Keep track of toroidal id 'i'?
            phi_range_i = (phi_range + 360*i/periodicity) % 360
            if phi_range_i.sum() == 0:
                # Account for 360%0=0
                phi_range_i[1] = 360
            mask_r = ((r_im >= r_range[0]) & (r_im <= r_range[1]))
            mask_phi = ((phi_im >= phi_range_i[0]) & (phi_im <= phi_range_i[1]))
            mask_z = ((z_im >= z_range[0]) & (z_im <= z_range[1]))
            if row['mirror_z']:
                # Reflect structure in z=0 plane for up down symetric components
                mask_z += ((z_im >= -z_range[1]) & (z_im <= -z_range[0]))
            mask = (mask_r & mask_phi & mask_z)

        overlapping_ids_mask = (~np.isnan(structure_ids))*mask if np.isnan(bg_value) else (structure_ids!=bg_value)*mask
        overlapping_ids = set(structure_ids[overlapping_ids_mask])
        if len(overlapping_ids) > 0:
            overlapping_structures = [visible_structures[i] for i in overlapping_ids]
            logger.warning(f'Surface coordinates overlap for surface "{surface_name}" ({surface_id}) and surface(s): '
                           f'{", ".join(overlapping_structures)}. '
                           f'Giving precedence to previously assigned structures: {overlapping_structures}.')
            mask.values[overlapping_ids_mask] = False
        if np.any(mask) > 0:
            structure_ids[mask] = surface_id
            material_ids[mask] = material_id
            visible_structures[surface_id] = surface_name
            visible_materials[material_id] = material_name
    if len(visible_structures) == 0:
        raise ValueError(f'No surfaces identified in camera view')
    VisibleStructuresTuple = namedtuple('VisibleStructuresTuple',
                                        ('structure_ids', 'material_ids', 'visible_structures', 'visible_materials'))
    return VisibleStructuresTuple(structure_ids, material_ids, visible_structures, visible_materials)

def identify_tiles_in_r_range(coord_r_values, tile_coords):
    raise NotImplementedError

def segment_path_by_material():  # pragma: no cover
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

def angles_to_convention(angles, units_input='radians', units_output='degrees', negative_angles=False,
                         cast_scalars=True):
    """Convert angles between radians and degrees and from +/-180 deg range to 0-360 range

    Args:
        angles: Array of angles
        units_input: Units of input data (radians/degrees)
        units_output: Units angles should be converted to (radians/degrees)
        negative_angles: Whether angles should be in range  +/-180 deg or 0-360 range
        cast_scalars: If output is 0D array cast to scalar float
    Returns:

    """
    if units_input in ('radians', 'rad'):
        units_input = 'radians'
    if units_output in ('degrees', 'deg'):
        units_output = 'degrees'

    if units_input not  in ('radians', 'degrees'):
        raise ValueError(f'Invalid input angle units: {units_input}')
    if units_output not in ('radians', 'degrees'):
        raise ValueError(f'Invalid output angle units: {units_output}')

    if (units_input == 'radians') and (units_output == 'degrees'):
        angles = np.rad2deg(angles)
    elif (units_input == 'degrees') and (units_output == 'radians'):
        angles = np.deg2rad(angles)

    shift = 360 if (units_output == 'degrees') else 2 * np.pi
    if (not negative_angles):
        angles = np.where(angles < 0, angles+shift, angles)
    else:
        half_circle = 180 if (units_output == 'degrees') else np.pi
        angles = np.where(angles > half_circle, angles-shift, angles)

    if cast_scalars:
        angles = ndarray_0d_to_scalar(angles)

    return angles


def cartesian_to_toroidal(x, y, z=None, angles_in_deg=False, angles_positive=True):
    """Convert cartesian coordinates to toroidal coordinates

    Args:
            x           : x cartesian coordinate(s)
            y           : y cartesian coordinate(s)
            z           : (Optional) z cartesian coordinate(s) - not used
        angles_in_deg  : Whether to convert phi output from radians to degrees
        angles_positive: Whether phi values should be in range positive [0, 360]/[[0,2pi] else [-180, +180]/[-pi, +pi]

    Returns: (r, phi, theta)

    """
    #TODO: Update call signature to be more inline with angles_to_convention()?
    r = np.hypot(x, y)
    phi = np.arctan2(y, x)  # Toroidal angle 'ϕ'

    # Poloidal angle 'ϑ'
    if z is not None:
        theta = np.arctan2(z, r)
    else:
        theta = np.full_like(x, np.nan)

    units = 'degrees'*angles_in_deg + 'radians' * (not angles_in_deg)
    phi = angles_to_convention(phi, units_input='radians', units_output=units, negative_angles=(not angles_positive))
    theta = angles_to_convention(theta, units_input='radians', units_output=units, negative_angles=(not angles_positive))

    return r, phi, theta

def cylindrical_to_cartesian(r, phi, z=None, angles_units='radians'):
    """Convert (R, phi) coords to (x, y) cartesian coords

    Args:
        r: Radial coordinate
        phi: Phi toroidal angle coordinate
        z: Z cartesian coordinate (ignored, but included so signature accepts full cylindrical coords)
        angles_units: Units of phi (radians/degrees)

    Returns: (Array of x coordinates, Array of y coordinates)

    """
    phi = angles_to_convention(phi, units_input=angles_units, units_output='radians', negative_angles=True)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y

def calc_horizontal_path_anulus_areas(r_path):
    """Return areas of horizontal annuli around machine at each point along the analysis path.

    Calculates the toroidal sum of horizontal surface area of tile around the machine between each radial point
    around the machine.
    Post processing corrections are required to account for the fact that the tiles surfaces are not actually
    horizontal due to both tilt in poloidal plane and toroidal tilt of tile (i.e. tile surface hight is func of
    toroidal angle z(phi))
     R1  R2  R3  R4  R.  Rn
    -|_._|_._|_._|_._|_._|-
    <-> <-> <-> <-> <-> <->
    dRf dRi ...         dRl

    Except for first/last boundaries:
        dR_i = [R_(i+1) - R_(i-1)] / 2

    The area of each horizontal annulus is:
        dA_i = 2 * pi * R * dR_i

    In MAST the horizontal divertor simplified the final area calculation to:
        2 * pi * R * dR

    Args:
        r_path: Radial coordinates of each point along the analysis path

    Returns: Areas of horizontal annuli around machine at each point along the analysis path.

    """
    r_path = np.array(r_path)
    dr = (r_path[2:] - r_path[0:-2]) / 2
    # As missing boundaries, set end differences to be one sided
    dr_first = [r_path[1]-r_path[0]]
    dr_last = [r_path[-1]-r_path[-2]]
    dr = np.abs(np.concatenate([dr_first, dr, dr_last]))
    annulus_areas = 2 * np.pi * r_path * dr
    # IDL comment: why not 1 / 2 as one Rib group only?
    # TF: not sure what Rib is?
    return annulus_areas


def calc_tile_tilt_area_correction_factors(path_data, poloidal_plane_tilt, nlouvres, step_size, path='path0'):
    """Return correction factors for areas returned by calc_horizontal_path_anulus_areas() accounting for tile tilts.

    See Matthew Dunn's MAST wetted area correction in:
    fire/misc/wetted_area_mdunn.py

    NOTES:
        - On MAST Wetted area fraction correction was independent of tile width - only depended on tile shadowing tilt
            and B field angle
        - On MAST-U angle is not constant and calculation requires tile width
        - Tilt of tiles in poloidal plane will affect wetted area
        - In MAST typical wetted fraction was ~0.7
        - In MAST-U early results suggest ~0.9
        - In MAST-U greater radial movement/width of leg will presumably make theta (B-field angle) radially dependent

    Wetted w_frac  = sin(θ)/sin(θ+α)
    sin(θ) =−B_z/√(B_tor^2+B_R^2+B_z^2 )
    tan(α) = (s*n)/(2*pi*R)

    Args:
        poloidal_plane_tilt: Angles of tile tilts in degrees relative to horizontal in poloidal plane
        nlouvres: Number of separate inclined tiles toroidally around the machine - requried to account for inter-tile
                    shadowing
        step_size: perpendicular step between adjacent tiles

    Returns: Multiplicative factors for annulus areas

    """
    tile_numbers = path_data[f'surface_id_in_frame_{path}']
    #correction_factor = np.full_like(tile_numbers, np.nan, dtype=float)
    n_pixels = path_data[f'i_{path}']
    #correction_factor = np.full_like(n_pixels, np.nan, dtype=float)
    correction_factor = np.full_like(path_data.heat_flux_path0.data, np.nan, dtype=float)
    factor_this_time = np.full_like(n_pixels, 1, dtype=float)
    shot = path_data.attrs['meta']['shot']
    #print('shot', shot)

    # Choose time point when most energy is deposited
    # The wetted fraction actually varies with time, which is especially important if there is a disruption
    #TODO implement time resolution to match EFIT
    #time = np.argmax(np.mean(path_data[f'heat_flux_{path}'], axis=1))
    #time = time.data.tolist()
    #print('time frame', time)
    #time = path_data.t[time]
    #time = time.data.tolist()
    #print('time value', time)

    # See which times EFIT converged for
    shotFile = f'/common/uda-scratch/lkogan/efitpp_eshed/epm0{shot}.nc'
    all_times = client.get('/epm/time', shotFile).data
    converged_status = client.get('/epm/equilibriumStatusInteger', shotFile)
    ind_converged = np.where(converged_status.data == 1)
    converged_times = all_times[ind_converged]

    #try:
    endtime = 1
    for i, time in enumerate(path_data.t.data):
        #print(f'time {time:.5f}')
        # Some if the times in the data have a rounding error, so are not quite equal to the EFIT times unless rounded
        if np.any(converged_times==float(f'{time:.5f}')):
            print(f'converged time {time:.5f}')


            #start = clock.time()
            # Call magnetic data from pyEquilibrium
            # If using the scheduler equilibrium for this shot
            #eqData = equilibrium('MAST', shot, time)
            # Or if using your own EFIT equilibrium
            #eqData = equilibrium(gfile=f'/home/mdunn/Documents/efit/{shot}/g_p{shot}_t{time:.5f}')
            # Or if using Lucy Kogan's epm files
            eqData = equilibrium(shot=shotFile, device='MASTU', time=time)
            B_Z = eqData.BZ
            B_T = eqData.Bt
            B_R = eqData.BR

            for pixel in path_data[f'i_{path}']:
                # Differentiate between tiles
                #tile_mask = tile_numbers == tile_number
                tile_number = tile_numbers.data[pixel]
                if tile_number > 0:
                    tile_name = f'T{tile_number}'

                    tile_step_size = step_size[tile_name]
                    tile_poloidal_plane_tilt = poloidal_plane_tilt[tile_name]  # In degrees
                    tile_nlouvres = nlouvres[tile_name]

                    # Mean spatial Coordinate of visible portion of this tile
                    # Consider increasing spatial resolution or implementing strike point tracking in future
                    radius = path_data[f'R_{path}'].data[pixel]
                    zCoord = path_data[f'z_{path}'].data[pixel]

                    # Magnetic field data at this point in space
                    bz = B_Z(radius, zCoord)[0][0]
                    bt = B_T(radius, zCoord)[0][0]
                    br = B_R(radius, zCoord)[0][0]

                    # Determine field angle
                    B_perp = bz*np.cos(tile_poloidal_plane_tilt*2*np.pi/360) + br*np.sin(tile_poloidal_plane_tilt*2*np.pi/360)
                    beta = np.arcsin(-B_perp/np.sqrt(bz**2 + bt**2 + br**2))
                    alpha = np.arctan(tile_step_size*tile_nlouvres/(2*np.pi*radius))  # In radians

                    # Set to 1 (one) to ignore
                    #correction_factor[tile_mask] = 1
                    #print(tile_name, 'f_wet')
                    #print(np.sin(beta)/np.sin(beta + alpha))
                    factor_this_time[pixel] = np.sin(beta)/np.sin(beta + alpha)
                    #print(correction_factor[tile_mask])
        correction_factor[i] = factor_this_time
        #end = clock.time()
        #print('time taken [s]:', end - start)

        # If this shot is not in the scheduler
        #except cpyuda.ServerException:
        #    print(f'Warning: there is no scheduler equilibrium for shot {shot}, setting tile tilt correction factor to 1')
        #    correction_factor[:] = 1

    #print(correction_factor)
    return correction_factor


def calc_divertor_area_integrated_param(values, annulus_areas):
    """Return physics parameter integrated across the divertor surface area

    Args:
        values: 1D (R) or 2D (R, t) array of parameter values
        annulus_areas: Annulus areas for each radial (R) coordinate

    Returns: Integrated parameter value (for each time point if 2D input)

    """
    # TODO: Check for 1/2D input
    if (isinstance(values, xr.DataArray) and isinstance(annulus_areas, xr.DataArray)):
        # Integrate along labeled axis
        spatial_dim = annulus_areas.dims[-1]
        value_times_area = values * annulus_areas
        total = value_times_area.sum(spatial_dim)
    else:
        total = np.sum(values * annulus_areas, axis=0)
    return total

if __name__ == '__main__':
    pass