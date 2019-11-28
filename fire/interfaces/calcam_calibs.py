#!/usr/bin/env python

"""Functions for interfacing with Calcam


Created: 11-10-19
"""

import logging, time
from typing import Union, Sequence, Optional
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import skimage
import matplotlib.pyplot as plt

import calcam
from fire import fire_paths
from fire.utils import locate_file

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

PathList = Sequence[Union[Path, str]]

calcam_calib_dir = Path('~/calcam2/calibrations/').expanduser()

def get_calcam_calib(calcam_calib_fn, calcam_calib_path='~/calcam/calibrations/'):
    """Return calcam Calibration object for given calibration filename and path

    :param calcam_calib_fn: Calibration filename
    :param calcam_calib_path: Calcam calibrations path
    :return: Calibration object
    """
    try:
        # Calcam 2.0+
        from calcam import Calibration
        calcam_calib_path_fn = Path(calcam_calib_path).expanduser() / calcam_calib_fn
        if calcam_calib_path_fn.suffix != '.ccc':
            calcam_calib_path_fn = calcam_calib_path_fn.with_suffix(calcam_calib_path_fn.suffix + '.ccc')
        calcam_calib = Calibration(calcam_calib_path_fn)
    except ImportError as e:
        # Calcam 1
        from calcam import fitting
        calcam_calib = fitting.CalibResults(calcam_calib_fn)
    return calcam_calib

def get_calcam_calib_info(pulse: int, camera: str, machine: str, search_paths: Optional[PathList]=None,
                          raise_=True) -> pd.Series:
    """Return path to calcam calibration file for given discharge and camera

    :param pulse: Shot/pulse number
    :param camera: Camera to look up calbration for
    :param machine: Tokamak under analysis
    :return: calib_path_fn
    """
    # TODO: Handle synthetic pulses with non numeric pulse numbers
    if search_paths is None:
        machine_inputs_path = fire_paths['input_files'] / f'{machine.lower()}'
        user_inputs_path = fire_paths['user_inputs'] / f'{machine.lower()}'
        search_paths = [machine_inputs_path, user_inputs_path]
    calib_lookup_fn = f'calcam_calibs-{machine.lower()}-{camera.lower()}-defaults.csv'

    calib_lookup_path, calib_lookup_fn = locate_file(search_paths, fns=calib_lookup_fn)
    calib_info = lookup_pulse_row_in_csv(Path(calib_lookup_path) / calib_lookup_fn, pulse)
    if raise_ and isinstance(calib_info, Exception):
        raise calib_info
    return calib_info

def apply_frame_display_transformations(frame_data, calcam_calib, image_coords):
    """Apply transformations (rotaions, reflections etc) to map camera frames from Original to Display coordinates

    Args:
        frame_data      : Frame data ndarray in "Original" coordinates with dimensions (x_pix, y_pix, frame_no)
        calcam_calib    : Calcam calibration object
        image_coords    : Coordinates to map to (either Original or Display). If original, no transformations are
                          applied

    Returns: np.ndarray with Display image transformations applied

    """
    # TODO: Implement conversion back to 'Original' Coordinates
    if image_coords == 'Display':
        # frame_data = np.moveaxis(frame_data, [0, 1, 2], [2, 1, 0])
        frame_data = np.moveaxis(frame_data, [0, 1, 2], [2, 0, 1])
        frame_data = calcam_calib.geometry.original_to_display_image(frame_data)
        # frame_data = np.moveaxis(frame_data, [0, 1, 2], [2, 1, 0])
        frame_data = np.moveaxis(frame_data, 2, 0)
    else:
        if image_coords != 'Original':
            raise ValueError(f'Unexpected value for "image_coords"="{image_coords}". Options are "Display" or '
                             f'"Original')
    return frame_data

# def get_calcam_calib_path_fn(calcam):
#     calcam_calib_path_fn = locate_file()
#
#     #####
#     calib_lookup_dflt_path_fn =  machine_inputs_path / calib_lookup_fn
#     calib_lookup_user_path_fn = user_inputs_path / calib_lookup_fn
#
#     path_fn = calib_lookup_user_path_fn
#     calib_info = lookup_pulse_row_in_csv(path_fn, pulse)
#     if isinstance(calib_info, Exception):
#         path_fn = calib_lookup_dflt_path_fn
#         calib_info = lookup_pulse_row_in_csv(path_fn, pulse)
#     if isinstance(calib_info, Exception):
#         raise calib_info
#     try:
#         calcam_calib = calib_info['calcam_calibration_file']
#     except KeyError as e:
#         raise ValueError(f'Calcam calib lookup file does not contain column "calcam_calibration_file": {path_fn}')
#     return calcam_calib

def get_surface_coords(calcam_calib, cad_model, outside_vesel_ray_length=10, image_coords='Original'):
    if image_coords == 'Display':
        image_shape = calcam_calib.geometry.get_display_shape()
    else:
        image_shape = calcam_calib.geometry.get_original_shape()
    # Use calcam convention: image data is indexed [y, x], but image shape description is (nx, ny)
    x_pix = np.arange(image_shape[0])
    y_pix = np.arange(image_shape[1])
    data_out = xr.Dataset(coords={'x_pix': x_pix, 'y_pix': y_pix})

    # Get wireframe image of CAD from camera view
    cad_model.set_wireframe(True)
    cad_model.set_colour((1, 0, 0))
    wire_frame = calcam.render_cam_view(cad_model, calcam_calib, coords=image_coords)

    logger.debug(f'Getting surface coords...'); t0 = time.time()
    ray_data = calcam.raycast_sightlines(calcam_calib, cad_model, coords=image_coords)
    print(f'Setup CAD model and cast rays in {time.time()-t0:1.1f} s')

    surface_coords = ray_data.get_ray_end(coords=image_coords)
    ray_lengths = ray_data.get_ray_lengths(coords=image_coords)
    mask_open_rays = np.where(ray_lengths > outside_vesel_ray_length)
    surface_coords[mask_open_rays[0], mask_open_rays[1], :] = np.nan
    ray_lengths[mask_open_rays] = np.nan

    data_out['x_im'] = (('y_pix', 'x_pix'), surface_coords[:, :, 0])
    data_out['y_im'] = (('y_pix', 'x_pix'), surface_coords[:, :, 1])
    data_out['z_im'] = (('y_pix', 'x_pix'), surface_coords[:, :, 2])
    data_out['R_im'] = (('y_pix', 'x_pix'), np.linalg.norm(surface_coords[:, :, 0:2], axis=2))
    data_out['phi_im'] = (('y_pix', 'x_pix'), np.arctan2(data_out['y_im'], data_out['x_im']))  # Toroidal angle 'ϕ'
    data_out['theta_im'] = (('y_pix', 'x_pix'), np.arctan2(data_out['z_im'], data_out['R_im']))  # Poloidal angle 'ϑ'
    data_out['ray_lengths'] = (('y_pix', 'x_pix'), ray_lengths)  # Distance from camera pupil to surface
    # Just take red channel of wireframe image
    data_out['wire_frame'] = (('y_pix', 'x_pix'), wire_frame[:, :, 0])
    # TODO: call plugin function to get sector, louvre and tile values
    # TODO: call plugin function to get s coordinate along tiles?
    return data_out

def project_analysis_path(raycast_data, analysis_path_dfn, calcam_calib):
    import matplotlib.pyplot as plt  # tmp
    # TODO: Handle combining multiple analysis paths?
    image_shape = np.array(calcam_calib.geometry.get_display_shape())
    points = pd.DataFrame.from_dict(list(analysis_path_dfn.values())[0], orient='index')
    points = points.rename(columns={'R': 'R_path_dfn', 'phi': 'phi_path_dfn', 'z': 'z_path_dfn'})
    pos_key = 'position'
    points.index.name = pos_key
    points = points.sort_values('order').to_xarray()
    phi_rad = np.deg2rad(points['phi_path_dfn'])
    points['x_path_dfn'] = points['R_path_dfn'] * np.cos(phi_rad)
    points['y_path_dfn'] = points['R_path_dfn'] * np.sin(phi_rad)

    points_xyz = points[['x_path_dfn', 'y_path_dfn', 'z_path_dfn']].to_array().T
    # Get image coordinates even if they are outside of the camera field of view
    points_pix = calcam_calib.project_points(points_xyz, fill_value=None)[0]
    points['x_pix_path_dfn'] = (points.coords, points_pix[0])
    points['y_pix_path_dfn'] = (points.coords, points_pix[1])

    pos_names = points.coords[pos_key]
    xpix_path, ypix_path, path_no = [], [], []
    for i_path, (start_pos, end_pos) in enumerate(zip(pos_names, pos_names[1:])):
        if not points['include_next_interval'].sel(position=start_pos):
            continue
        x0, y0, x1, y1 = np.round((*points['x_pix_path_dfn'].sel({pos_key: slice(start_pos, end_pos)}),
                                   *points['y_pix_path_dfn'].sel({pos_key: slice(start_pos, end_pos)}))).astype(int)
        # Use Bresenham's line drawing algorithm. npoints = max((dx, dy))
        xpix, ypix = skimage.draw.line(x0, y0, x1, y1)
        xpix_path.append(xpix)
        ypix_path.append(ypix)
        path_no.append(np.full_like(xpix, i_path))
    xpix_path = np.concatenate(xpix_path)
    ypix_path = np.concatenate(ypix_path)
    path_no = np.concatenate(path_no)

    analysis_path = xr.Dataset(coords={'i_path': ('path', np.arange(len(xpix))), 'segment': ('path', path_no)})
    analysis_path['x_pix_path'] = (('i_path',), xpix_path)
    analysis_path['y_pix_path'] = (('i_path',), ypix_path)
    analysis_path['visible_path'] = (('i_path',), check_visible(xpix_path, ypix_path, image_shape[::-1]))
    index_path = {'x_pix': xr.DataArray(xpix_path, dims='i_path'),
                  'y_pix': xr.DataArray(ypix_path, dims='i_path')}
    for coord in ['R', 'phi', 'x', 'y', 'z']:
        analysis_path[coord+'_path'] = (('i_path',), raycast_data[coord+'_im'].sel(index_path))

    # TODO: check_occlusion
    if np.any(~analysis_path['visible_path']):
        logger.warning(f'Analysis path contains sections that are not visible from the camera: '
                       f'{~analysis_path["visible_path"]}')
    return analysis_path

def check_visible(x_points, y_points, image_shape):
    # TODO: Check calcam convension for subwindows
    x_points, y_points = np.array(x_points), np.array(y_points)
    visible = (x_points >= 0) & (x_points < image_shape[1]) & (y_points >= 0) & (y_points < image_shape[0])
    return visible.astype(bool)

if __name__ == '__main__':
    pass