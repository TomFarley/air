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

import calcam
from fire import fire_paths
from fire.geometry.geometry import cartesian_to_toroidal
from fire.camera.image_processing import find_outlier_intensity_threshold
from fire.interfaces.interfaces import lookup_pulse_row_in_csv
from fire.misc.utils import locate_file, make_iterable, to_image_dataset

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

def get_surface_coords(calcam_calib, cad_model, outside_vesel_ray_length=10, image_coords='Original',
                       phi_positive=True, remove_long_rays=True):
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
    # open rays that don't terminate in the vessel
    mask_bad_data = ray_lengths > outside_vesel_ray_length
    if remove_long_rays:
        thresh_long_rays = find_outlier_intensity_threshold(ray_lengths)
        mask_long_rays = ray_lengths > thresh_long_rays
        mask_bad_data += mask_long_rays
    ind_bad_data = np.where(mask_bad_data)
    surface_coords[ind_bad_data[0], ind_bad_data[1], :] = np.nan
    ray_lengths[mask_bad_data] = np.nan
    if len(ind_bad_data[0]) > 0:
        logger.info(f'Spatial coords for {len(ind_bad_data[0])} pixels set to "nan" due to holes in CAD model')

    x = surface_coords[:, :, 0]
    y = surface_coords[:, :, 1]
    z = surface_coords[:, :, 2]
    r, phi, theta = cartesian_to_toroidal(x, y, z, angles_in_deg=False, angles_positive=phi_positive)

    data_out['x_im'] = (('y_pix', 'x_pix'), x)
    data_out['y_im'] = (('y_pix', 'x_pix'), y)
    data_out['z_im'] = (('y_pix', 'x_pix'), z)
    data_out['R_im'] = (('y_pix', 'x_pix'), r)
    data_out['phi_im'] = (('y_pix', 'x_pix'), phi)  # Toroidal angle 'ϕ'
    data_out['phi_deg_im'] = (('y_pix', 'x_pix'), np.rad2deg(phi))  # Toroidal angle 'ϕ' in degrees
    data_out['theta_im'] = (('y_pix', 'x_pix'), theta)
    data_out['ray_lengths_im'] = (('y_pix', 'x_pix'), ray_lengths)  # Distance from camera pupil to surface
    data_out['bad_cad_coords_im'] = (('y_pix', 'x_pix'), mask_bad_data.astype(int))  # Pixels seeing holes in CAD model
    spatial_res = calc_spatial_res(x, y, z, res_min=1e-4, res_max=None)
    for key, value in spatial_res.items():
        data_out[key] = (('y_pix', 'x_pix'), value)
    # Add labels for plots
    data_out['spatial_res_max'].attrs['standard_name'] = 'Spatial resolution'
    data_out['spatial_res_max'].attrs['units'] = 'm'
    # Just take red channel of wireframe image
    # TODO: Fix wireframe image being returned black
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

def project_analysis_path(raycast_data, analysis_path_dfn, calcam_calib, path_name, masks=None):
    """Project an analysis path defined by a set of spatial coordinates along tile surfaces into camera image coords

    Args:
        raycast_data: Dataset of spatial coordinate info for each pixel in camera images
        analysis_path_dfn: Spatial coordinates of points defining analysis path
        calcam_calib: Calcam calibration object
        path_name: Name of path (short name) used in DataArray variable and coordiante names
        masks: (optional) Additional image data to extract along path

    Returns: Dataset of variables defining analysis_path through image (eg. x_pix, y_pix etc.)

    """
    path = path_name  # abbreviation for format strings
    path_coord = f'i_{path}'  # xarray index coordinate along analysis path

    # TODO: Handle combining multiple analysis paths? Move loop over paths below to here/outside fuction...?
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
    # TODO: Handle points outside field of view

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
            try:
                mask_data = masks[key][ypix, xpix]
            except IndexError as e:
                from fire.plotting.image_figures import figure_analysis_path
                logger.error(f'Analysis path strays outside image - not currently supported')
                data = to_image_dataset(masks[key], key)
                xpix_path = np.concatenate(xpix_path)
                ypix_path = np.concatenate(ypix_path)
                data[path_coord] = (path_coord, np.arange(len(xpix_path)))
                data[f'x_pix_{path}'] = (path_coord, xpix_path)
                data[f'y_pix_{path}'] = (path_coord, ypix_path)
                figure_analysis_path(data, key=key, show=True)
                raise
            masks_path[key].append(mask_data)
    xpix_path = np.concatenate(xpix_path)
    ypix_path = np.concatenate(ypix_path)
    path_no = np.concatenate(path_no)
    for key in masks_path:
        masks_path[key] = np.concatenate(masks_path[key])

    # NOTE: path substitutions below are currently over generalised?
    analysis_path = xr.Dataset()
    coords = {f'i_{path}': (f'i_{path}', np.arange(len(xpix_path)))}  #,
              # f'segment_{path}': (f'{path}', path_no)}

    analysis_path = analysis_path.assign_coords(**coords)
    analysis_path[f'segment_{path}'] = ((f'i_{path}',), path_no)
    analysis_path[f'y_pix_{path}'] = ((f'i_{path}',), ypix_path)
    analysis_path[f'x_pix_{path}'] = ((f'i_{path}',), xpix_path)
    analysis_path[f'visible_{path}'] = ((f'i_{path}',), check_visible(xpix_path, ypix_path, image_shape[::-1]))
    index_path = {'x_pix': xr.DataArray(xpix_path, dims=f'i_{path}'),
                  'y_pix': xr.DataArray(ypix_path, dims=f'i_{path}')}
    for coord in ['R', 'phi', 'x', 'y', 'z']:
        analysis_path[coord+f'_{path}'] = ((f'i_{path}',), raycast_data[coord+'_im'].sel(index_path))
    for key in masks_path:
        analysis_path[key+f'_{path}'] = ((f'i_{path}',), masks_path[key])

    # TODO: check_occlusion
    if np.any(~analysis_path[f'visible_{path}']):
        logger.warning(f'Analysis path contains sections that are not visible from the camera: '
                       f'{~analysis_paths["visible_{path}"]}')
    return analysis_path

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
        if str(e) == "module 'calcam' has no attribute 'CADModel'":
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