# -*- coding: future_fstrings -*-
#!/usr/bin/env python

"""Functions for interfacing with Calcam


Created: 11-10-19
"""

import logging, time
from typing import Union, Sequence, Optional
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import xarray as xr
import skimage
import matplotlib.pyplot as plt

import calcam
from fire import fire_paths
from fire.utils import locate_file, make_iterable
from fire.interfaces.interfaces import lookup_pulse_row_in_csv

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

PathList = Sequence[Union[Path, str]]

module_default = object()  # sentinel

pwd = Path(__file__).parent
calcam_calib_dir = Path('~/calcam2/calibrations/').expanduser()
fire_cad_dir = (pwd  / '../input_files/cad/').resolve()

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
    data_out['phi_deg_im'] = (('y_pix', 'x_pix'), np.rad2deg(data_out['phi_im']))  # Toroidal angle 'ϕ' in degrees
    data_out['theta_im'] = (('y_pix', 'x_pix'), np.arctan2(data_out['z_im'], data_out['R_im']))  # Poloidal angle 'ϑ'
    data_out['ray_lengths'] = (('y_pix', 'x_pix'), ray_lengths)  # Distance from camera pupil to surface
    spatial_res = calc_spatial_res(data_out['x_im'], data_out['x_im'], data_out['x_im'])
    for key, value in spatial_res.items():
        data_out[key] = (('y_pix', 'x_pix'), value)
    # Add labels for plots
    data_out['spatial_res_max'].attrs['standard_name'] = 'Spatial resolution'
    data_out['spatial_res_max'].attrs['units'] = 'm'
    # Just take red channel of wireframe image
    data_out['wire_frame'] = (('y_pix', 'x_pix'), wire_frame[:, :, 0])

    return data_out

def calc_spatial_res(x_im, y_im, z_im, res_min=1e-4, res_max=None):
    """Return spatial resolution at each pixel given cartesian spatial coords of each pixel

    Args:
        x_im: Array of cartesian x spatial coordinates (e.g. from calcam) for each pixel (in meters)
        y_im: Array of cartesian y spatial coordinates (e.g. from calcam) for each pixel (in meters)
        z_im: Array of cartesian z spatial coordinates (e.g. from calcam) for each pixel (in meters)
        res_min: Minimum resolution considered realistic (lower values are set to nan) eg 1e-4 (<0.1 mm)
        res_max: Maximum resolution considered realistic (higher values are set to nan) eg 5e-1 (>50 cm)

    Returns: Array containing spatial coords of each pixel

    """
    coords = (x_im, y_im, z_im)
    spatial_res = {}
    # Calculate spatial distance between adjacent pixels
    spatial_res_x = np.linalg.norm([np.diff(coord, axis=1) for coord in coords], axis=0)
    spatial_res_y = np.linalg.norm([np.diff(coord, axis=0) for coord in coords], axis=0)
    # Pad arrays to make them same shape as image
    spatial_res_x = np.pad(spatial_res_x, pad_width=((0, 0), (1, 0)), mode='edge')
    spatial_res_y = np.pad(spatial_res_y, pad_width=((1, 0), (0, 0)), mode='edge')
    # Remove unrealistically low/zero values eg res < 1e-4  (0.1 mm)
    spatial_res_x = np.where(spatial_res_x < res_min, np.nan, spatial_res_x)
    spatial_res_y = np.where(spatial_res_y < res_min, np.nan, spatial_res_y)
    if res_max is not None:
        # Remove unrealistically high values eg res > 5e-1  (50 cm)
        spatial_res_x = np.where(spatial_res_x > res_max, np.nan, spatial_res_x)
        spatial_res_y = np.where(spatial_res_y > res_max, np.nan, spatial_res_y)
    else:
        # Remove extreme values between surfaces at different distances
        spatial_res_x = np.where(spatial_res_x > np.nanmean(spatial_res_x)+3.5*np.nanstd(spatial_res_x), np.nan,
                                 spatial_res_x)
        spatial_res_y = np.where(spatial_res_y > np.nanmean(spatial_res_y)+3.5*np.nanstd(spatial_res_y), np.nan,
                                 spatial_res_y)

    spatial_res['spatial_res_x'] = spatial_res_x
    spatial_res['spatial_res_y'] = spatial_res_y
    spatial_res['spatial_res_mean'] = np.nanmean([spatial_res_x, spatial_res_y], axis=0)
    spatial_res['spatial_res_max'] = np.nanmax([spatial_res_x, spatial_res_y], axis=0)

    return spatial_res

def project_analysis_path(raycast_data, analysis_path_dfn, calcam_calib, masks=None, debug=True):
    # TODO: Handle combining multiple analysis paths? Move loop over paths below to here...
    image_shape = np.array(calcam_calib.geometry.get_display_shape())
    # points = pd.DataFrame.from_dict(list(analysis_path_dfn.values())[0], orient='index')
    # points = pd.DataFrame.from_items(analysis_path_dfn).T
    points = pd.DataFrame.from_dict(OrderedDict(analysis_path_dfn)).T
    points = points.rename(columns={'R': 'R_path_dfn', 'phi': 'phi_path_dfn', 'z': 'z_path_dfn'})
    points = points.astype({'R_path_dfn': float, 'include_next_interval': bool, 'order': int, 'phi_path_dfn': float,
                           'z_path_dfn': float})
    # TODO: sort df point by 'order' column
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
    # x and y pixel value and path index number for each point along analysis path
    # Path index (path_no) indexes which pair of points in the path definition a given point along the path belongs to
    xpix_path, ypix_path, path_no = [], [], []
    masks_path = {key: [] for key in masks} if masks else {}
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
        for key in masks_path:
            mask_data = masks[key][ypix, xpix]
            masks_path[key].append(mask_data)
    xpix_path = np.concatenate(xpix_path)
    ypix_path = np.concatenate(ypix_path)
    path_no = np.concatenate(path_no)
    for key in masks_path:
        masks_path[key] = np.concatenate(masks_path[key])

    # TODO: Move loop to top of fucntion and pass multiple paths with different names?
    # NOTE: path substitutions below are currently over generalised
    paths = ['path']
    analysis_paths = xr.Dataset()
    for path in paths:
        coords = {f'i_{path}': (f'{path}', np.arange(len(xpix))),
                  f'segment_{path}': (f'{path}', path_no)}
        analysis_paths = analysis_paths.assign_coords(**coords)
        analysis_paths[f'y_pix_{path}'] = ((f'i_{path}',), ypix_path)
        analysis_paths[f'x_pix_{path}'] = ((f'i_{path}',), xpix_path)
        analysis_paths[f'visible_{path}'] = ((f'i_{path}',), check_visible(xpix_path, ypix_path, image_shape[::-1]))
        index_path = {'x_pix': xr.DataArray(xpix_path, dims=f'i_{path}'),
                      'y_pix': xr.DataArray(ypix_path, dims=f'i_{path}')}
        for coord in ['R', 'phi', 'x', 'y', 'z']:
            analysis_paths[coord+f'_{path}'] = ((f'i_{path}',), raycast_data[coord+'_im'].sel(index_path))
        for key in masks_path:
            analysis_paths[key+f'_{path}'] = ((f'i_{path}',), masks_path[key])

        # TODO: check_occlusion
        if np.any(~analysis_paths[f'visible_{path}']):
            logger.warning(f'Analysis path contains sections that are not visible from the camera: '
                           f'{~analysis_paths["visible_{path}"]}')
    return analysis_paths

def check_visible(x_points, y_points, image_shape):
    # TODO: Check calcam convension for subwindows
    x_points, y_points = np.array(x_points), np.array(y_points)
    visible = (x_points >= 0) & (x_points < image_shape[1]) & (y_points >= 0) & (y_points < image_shape[0])
    return visible.astype(bool)

def get_calcam_cad_obj(model_name, model_variant, check_fire_cad_defaults=True):
    logger.debug(f'Loading CAD model...');
    t0 = time.time()
    # TODO: Add error messages directing user to update Calcam CAD definitions in settings GUI if CAD not found
    print(dir(calcam))
    try:
        cad_model = calcam.CADModel(model_name=model_name, model_variant=model_variant)
    except ValueError as e:
        if check_fire_cad_defaults:
            add_calcam_cad_paths(required_models=model_name)
            cad_model = calcam.CADModel(model_name=model_name, model_variant=model_variant)
        else:
            raise
    except AttributeError as e:
        if str(e) == module 'calcam' has no attribute 'CADModel':
            logger.warning('Calcam failed to import calcam.cadmodel.CADModel presumably due to vtk problem')
            import cv2
            import vtk
        raise
    logger.debug(f'Setup CAD model object in {time.time()-t0:1.1f} s')
    return cad_model

def add_calcam_cad_paths(paths=module_default, required_models=None):
    """Update paths where calcam locates CAD .ccc files.

    Args:
        paths: Path(s) to directory containing CAD .ccc files
        required_models: Name or list of names of CAD models that should be locatable after path added

    Returns: Dictionary where the keys are the human-readable names of the machine models calcam has found,
             and the values are lists containing
             [ full path to the CAD file ,  Available model variants ,Default model variant ]
    """
    if paths is module_default:
        paths = fire_cad_dir
    conf = calcam.config.CalcamConfig()
    models = conf.get_cadmodels()
    n_models = len(models)

    for path in make_iterable(paths):
        cad_path = Path(path).resolve()
        conf.cad_def_paths.append(str(cad_path))
    conf.save()
    logger.debug(f'Added paths for Calcam CAD model lookup: {paths}')
    logger.debug(f'Resulting Calcam CAD model lookup paths: {conf.cad_def_paths}')

    if required_models is not None:
        missing_models = check_calcam_cad_found(required_models, raise_on_missing=True)

    if len(models) == n_models:
        logger.warning(f'Failed to locate new CAD models in {paths}')
    return models

def check_calcam_cad_found(required_models, raise_on_missing=False):
    conf = calcam.config.CalcamConfig()
    models = conf.get_cadmodels()
    missing_models = [model for model in make_iterable(required_models) if model not in models]
    if raise_on_missing and (len(missing_models) > 0):
        raise FileNotFoundError(f'Failed to locate CAD model for {missing_models}. Models can be added with:\n'
                                f'conf = calcam.config.CalcamConfig()\n'
                                f'conf.cad_def_paths.append(cad_path)')
    return missing_models

if __name__ == '__main__':
    pass