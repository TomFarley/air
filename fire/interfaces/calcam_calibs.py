# -*- coding: future_fstrings -*-
#!/usr/bin/env python

"""Functions for interfacing with Calcam

Excert from Calcam documentation:
Image Pixel Coordinates
 Calcam follows the convention of using matrix/linear algebra style pixel coordinates for images, which is consistent with the way images are stored and addressed in 2D arrays in Python. In this convention, the origin (0,0) is in the centre of the top-left pixel in the image. The y axis runs from top to bottom down the image and the x axis runs horizontally left to right.
 It is important to note that since 2D arrays are indexed [row, column], arrays containing images are indexed as [y,
 x]. However, Calcam functions which deal with image coordinates are called as function(x,y). This is consistent with the way image coordinates are delt with in OpenCV.

Resolution of calibration calibration can be returned with
 geom = calcam_calibration.geometry
 sensor_resolution = (geom.get_original_shape() if coords.lower() == 'original' else geom.get_display_shape())
This returns a tuple of (x_width, y_height)
A normal call to plt.imshow(frame) will display the image with the x axis running from left to right and the y axis
running from top to bottom.

Created: 11-10-19
"""

import logging, time
from typing import Union, Sequence, Optional
from pathlib import Path
from collections import OrderedDict, defaultdict
from copy import copy

import numpy as np
import pandas as pd
import xarray as xr
import skimage

import calcam
from fire import fire_paths
from fire.geometry.geometry import cartesian_to_toroidal, cylindrical_to_cartesian, angles_to_convention
from fire.camera.image_processing import find_outlier_intensity_threshold
from fire.interfaces.interfaces import lookup_pulse_row_in_csv
from fire.misc.data_quality import calc_outlier_nsigma_for_sample_size
from fire.misc.utils import locate_file, make_iterable, to_image_dataset

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

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

def update_detector_window(calcam_calib: calcam.Calibration, detector_window: Optional[Union[list,tuple]]=None,
                           frame_data: Optional[Union[xr.DataArray,np.ndarray]]=None, coords: str='Original'):
    """

    Args:
        calcam_calib:    Calcam Calibration object
        detector_window: (tuple or list) – A 4-element tuple or list of integers defining the detector window
                         coordinates (Left,Top,Width,Height)
        frame_data:      DataArray/ndarray of frame data
        coords:          Whether frame data is passed in calcam "Original" or "Display" coordinates

    Returns: calcam_calib, window_info

    """
    # TODO: Handle setting window when there are multiple subviews - need to extract subview windows from subview mask?
    geom = calcam_calib.geometry
    sensor_resolution = (geom.get_original_shape() if coords.lower() == 'original' else geom.get_display_shape())
    if (detector_window is None) or np.any(np.isnan(detector_window)):
        if frame_data is None:
            raise ValueError(f'Require either frame data or detector window as inputs. Both None.')
        logger.warning(f'Detector window not supplied - assuming centred detector window')
        image_resolution = frame_data.shape[:0:-1]  # (x_width, y_hight)
        # TODO: Check using x, y coords right way round with calcam conventions etc, +1 offsets etc
        left = int(sensor_resolution[0]/2 - image_resolution[0]/2)
        top = int(sensor_resolution[1]/2 - image_resolution[1]/2)  # Origin in top left
        detector_window = np.array([left, top, *image_resolution])  # +1
    else:
        # TODO: Check if image_resolution needs reversing?
        image_resolution = detector_window[2:]
    assert len(detector_window) == 4

    detector_window_applied = np.all(sensor_resolution == image_resolution)

    # NOTE: Detector window coordinates must always be in "original" coordinates.
    try:
        calcam_calib.set_detector_window(window=np.array(detector_window).astype(int))
    except Exception as e:
        # TODO: Work out how to divide up detector_window by sub-view and remove warning for unhandled exception
        if calcam_calib.n_subviews > 1:
            logger.warning(f'Setting detector window failed due to multiple subviews')
        else:
            raise e
    else:
        logger.info('Set calcam detector window to: %s (Left,Top,Width,Height)', detector_window)

    # NOTE: Calls to calcam_calib.geometry.get_original_shape() now return the windowed detector size

    window_info = dict(detector_window=np.array(detector_window), image_resolution=image_resolution,
                       sensor_resolution=sensor_resolution, detector_window_applied=detector_window_applied)
    return window_info


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

def get_surface_coords(calcam_calib, cad_model, image_coords='Original', phi_positive=True, intersecting_only=True,
                       exclusion_radius=0.10, remove_long_rays=True, outside_vesel_ray_length=10):
    if image_coords.lower() == 'display':
        image_shape = calcam_calib.geometry.get_display_shape()
    else:
        image_shape = calcam_calib.geometry.get_original_shape()
    # Use calcam convention: image data is indexed [y, x], but image shape description is (nx, ny)
    x_pix = np.arange(image_shape[0])
    y_pix = np.arange(image_shape[1])
    data_out = xr.Dataset(coords={'x_pix': x_pix, 'y_pix': y_pix})

    # Get wireframe image of CAD from camera view
    cad_model.set_flat_shading(False)  # lighting effects
    cad_model.set_wireframe(True)
    # cad_model.set_linewidth(3)
    color = cad_model.get_colour()
    # cad_model.set_colour((1, 0, 0))
    wire_frame = calcam.render_cam_view(cad_model, calcam_calib, coords=image_coords, verbose=False)

    logger.debug(f'Getting surface coords...'); t0 = time.time()

    ray_data = calcam.raycast_sightlines(calcam_calib, cad_model, coords=image_coords,
                                         exclusion_radius=exclusion_radius, intersecting_only=intersecting_only)
    # TODO: Set sensor subwindow if using full sensor calcam calibration for windowed view
    # ray_data.set_detector_window(window=(Left,Top,Width,Height))
    logger.debug(f'Setup CAD model and cast rays in {time.time()-t0:1.1f} s')

    surface_coords = ray_data.get_ray_end(coords=image_coords)
    ray_lengths = ray_data.get_ray_lengths(coords=image_coords)
    # open rays that don't terminate in the vessel - should no longer be required now calcam.raycast_sightlines has
    # the keyword intersecting_only=True
    mask_bad_data = ray_lengths > outside_vesel_ray_length
    if remove_long_rays and (not intersecting_only):
        nsigma = calc_outlier_nsigma_for_sample_size(ray_lengths.size)
        thresh_long_rays = find_outlier_intensity_threshold(ray_lengths, nsigma=nsigma)
        mask_long_rays = ray_lengths > thresh_long_rays
        mask_bad_data += mask_long_rays
    ind_bad_data = np.where(mask_bad_data)
    surface_coords[ind_bad_data[0], ind_bad_data[1], :] = np.nan
    ray_lengths[mask_bad_data] = np.nan
    if len(ind_bad_data[0]) > 0:
        logger.info(f'Spatial coords for {len(ind_bad_data[0])} pixels set to "nan" due to holes in CAD model')
        if intersecting_only:
            logger.warning(f'Excessively long rays (>{thresh_long_rays} m) despite intersecting_only=True')

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
    # Indices of sub calibrations due to mirrors etc
    data_out['subview_mask_im'] = (('y_pix', 'x_pix'), calcam_calib.get_subview_mask(coords=image_coords))
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

def project_spatial_analysis_path(raycast_data, analysis_path_dfn, calcam_calib, path_name, masks=None,
                                  image_coords='Display'):
    """Project an analysis path defined by a set of spatial coordinates along tile surfaces into camera image coords

    Args:
        raycast_data: Dataset of spatial coordinate info for each pixel in camera images
        analysis_path_dfn: Spatial coordinates of points defining analysis path
        calcam_calib: Calcam calibration object
        path_name: Name of path (short name) used in DataArray variable and coordiante names
        masks: (optional) Additional image data to extract along path

    Returns: Dataset of variables defining analysis_path through image (eg. x_pix, y_pix etc.)

    """
    # TODO: Split into two functions: one that projects the spatial path definition points onto image path defnintion
    #  points and a second that uses skimage.draw to connect up the image path definition points. This will allow old
    #  MAST image path definitions to be used for exact regression tests
    path = path_name  # abbreviation for format strings
    path_coord = f'i_{path}'  # xarray index coordinate along analysis path

    # TODO: Handle combining multiple analysis paths? Move loop over paths below to here/outside fuction...?
    subview_mask = calcam_calib.get_subview_mask(coords=image_coords)
    image_shape = np.array(subview_mask.shape)
    # points = pd.DataFrame.from_dict(list(analysis_path_dfn.values())[0], orient='index')
    # points = pd.DataFrame.from_items(analysis_path_dfn).T
    points = pd.DataFrame.from_dict(OrderedDict(analysis_path_dfn)).T
    points = points.rename(columns={'R': f'R_{path}_dfn', 'phi': f'phi_{path}_dfn', 'z': f'z_{path}_dfn'})
    points = points.astype({f'R_{path}_dfn': float, 'include_next_interval': bool, 'order': int, f'phi_{path}_dfn': float,
                           f'z_{path}_dfn': float})
    # TODO: sort df point by 'order' column
    pos_key = 'position'
    points.index.name = pos_key
    points = points.sort_values('order').to_xarray()
    phi_rad = np.deg2rad(points[f'phi_{path}_dfn'])
    points[f'x_{path}_dfn'] = points[f'R_{path}_dfn'] * np.cos(phi_rad)
    points[f'y_{path}_dfn'] = points[f'R_{path}_dfn'] * np.sin(phi_rad)

    points_xyz = points[[f'x_{path}_dfn', f'y_{path}_dfn', f'z_{path}_dfn']].to_array().T
    # Get image coordinates even if they are outside of the camera field of view
    # NOTE: Calcam.project_points returns a list of image (x,y) coordinates for each subview. Images are indexed [y, x].
    points_pix_subviews = calcam_calib.project_points(points_xyz, fill_value=None)
    points_pix, info = select_visible_points_from_subviews(points_pix_subviews, subview_mask, points_xyz=points_xyz,
                                                     subviews_keep='all',
                                                     raise_on_duplicate_view=True, raise_on_out_of_frame=True)
    points_pix = points_pix.astype(int)
    points[f'{path}_dfn_x_pix'] = (points.coords, points_pix[0])
    points[f'{path}_dfn_y_pix'] = (points.coords, points_pix[1])
    points[f'{path}_dfn_subview'] = (points.coords, info['subview_origin'])
    points[f'{path}_dfn_visible'] = (points.coords, info['mask_visible'])

    pos_names = points.coords[pos_key]
    # x and y pixel value and path index number for each point along analysis path
    # Path index (path_no) indexes which pair of points in the path definition a given point along the path belongs to
    xpix_path, ypix_path, path_no, xpix_out_of_frame, ypix_out_of_frame = [], [], [], [], []
    masks_path = {key: [] for key in masks} if masks else {}
    for i_path, (start_pos, end_pos) in enumerate(zip(pos_names, pos_names[1:])):
        if not points['include_next_interval'].sel(position=start_pos):
            continue
        x0, y0, x1, y1 = np.round((*points[f'{path}_dfn_x_pix'].sel({pos_key: slice(start_pos, end_pos)}),
                                   *points[f'{path}_dfn_y_pix'].sel({pos_key: slice(start_pos, end_pos)}))).astype(int)
        # Use Bresenham's line drawing algorithm. npoints = max((dx, dy))
        xpix_all, ypix_all = skimage.draw.line(x0, y0, x1, y1)

        # Check if path strays outside image
        mask_in_frame = check_in_frame(xpix_all, ypix_all, image_shape)
        if np.any(~mask_in_frame):
            from fire.plotting.image_figures import figure_analysis_path
            logger.error(f'Some points defining the analysis path "{path_name}" stray outside the image')
            data = to_image_dataset(masks[key], key)
            xpix_path = np.concatenate(xpix_path)
            ypix_path = np.concatenate(ypix_path)
            data[path_coord] = (path_coord, np.arange(len(xpix_path)))
            data[f'x_pix_{path}'] = (path_coord, xpix_path)
            data[f'y_pix_{path}'] = (path_coord, ypix_path)
            figure_analysis_path(data, key=key, show=True)
            # raise

        # Just keep parts of path that are visible
        xpix = xpix_all[mask_in_frame]
        ypix = ypix_all[mask_in_frame]
        xpix_path.append(xpix)
        ypix_path.append(ypix)
        path_no.append(np.full_like(xpix, i_path))

        # Also record parts of path outside of frame for plotting/debugging etc
        xpix_out_of_frame.append(xpix_all[~mask_in_frame])
        ypix_out_of_frame.append(ypix_all[~mask_in_frame])

        # Extract values along path from data masks
        for key in masks_path:
            mask_data = masks[key][ypix, xpix]
            masks_path[key].append(mask_data)

    xpix_path = np.concatenate(xpix_path)
    ypix_path = np.concatenate(ypix_path)
    path_no = np.concatenate(path_no)
    xpix_out_of_frame = np.concatenate(xpix_out_of_frame)
    ypix_out_of_frame = np.concatenate(ypix_out_of_frame)
    for key in masks_path:
        masks_path[key] = np.concatenate(masks_path[key])

    # NOTE: path substitutions below are currently over generalised?
    analysis_path = xr.Dataset()
    coords = {f'i_{path}': (f'i_{path}', np.arange(len(xpix_path)))}
    coords_oof = {f'i_{path}_out_of_frame': (f'i_{path}_out_of_frame', np.arange(len(xpix_out_of_frame)))}
    coords.update(coords_oof)

    analysis_path = analysis_path.assign_coords(**coords)
    # TODO: Change to correct UDA units for arb array index
    analysis_path[f'i_{path}'].attrs.update(dict(units='count',
                                                 label=f'Array index along analysis path "{path}" through IR image'))
    # TODO: Add labels and units to other coords and vars written to UDA
    analysis_path[f'segment_{path}'] = ((f'i_{path}',), path_no)
    analysis_path[f'y_pix_{path}'] = ((f'i_{path}',), ypix_path)
    analysis_path[f'x_pix_{path}'] = ((f'i_{path}',), xpix_path)
    # Include pixel coords of path elements outside of the frame
    analysis_path[f'y_pix_{path}_out_of_frame'] = ((f'i_{path}_out_of_frame',), ypix_out_of_frame)
    analysis_path[f'x_pix_{path}_out_of_frame'] = ((f'i_{path}_out_of_frame',), xpix_out_of_frame)
    # Calcam uses convention that the origin (0,0) is in the centre of the top-left pixel - reverse y axis for indexing?
    subview_path = subview_mask[::-1, :][ypix_path, xpix_path]
    analysis_path[f'subview_{path}'] = ((f'i_{path}',), subview_path)  # Which subview each pixel is from
    # analysis_path[f'in_frame_{path}'] = ((f'i_{path}',), check_in_frame(xpix_path, ypix_path, image_shape[::-1]))
    index_path = {'x_pix': xr.DataArray(xpix_path, dims=f'i_{path}'),
                  'y_pix': xr.DataArray(ypix_path, dims=f'i_{path}')}
    for coord in ['R', 'phi', 'x', 'y', 'z']:
        analysis_path[coord+f'_{path}'] = ((f'i_{path}',), raycast_data[coord+'_im'].sel(index_path))
    for key in masks_path:
        analysis_path[key+f'_{path}'] = ((f'i_{path}',), masks_path[key])

    # TODO: check_occlusion
    if len(xpix_out_of_frame) > 0:
        logger.warning(f'Analysis path contains sections that are not in frame: {len(xpix_out_of_frame)} points')
    return analysis_path

def check_in_frame(x_points, y_points, image_shape):
    # TODO: Check calcam convension for subwindows
    # NOTE: Calcam.project_points returns a list of image (x,y) coordinates for each subview. Images are indexed [y, x].
    # TODO: check for alpha channel or frame number dim
    x_points, y_points = np.array(x_points), np.array(y_points)
    visible = (x_points >= 0) & (x_points < image_shape[1]) & (y_points >= 0) & (y_points < image_shape[0])
    return visible.astype(bool)

def cartesian_to_pixel_coordinates(calcam_calib, points_xyz, image_coords='Display', raise_on_out_of_frame=True):
    subview_mask = calcam_calib.get_subview_mask(coords=image_coords)

    # Get image coordinates even if they are outside of the camera field of view
    # NOTE: Calcam.project_points returns a list of image (x,y) coordinates for each subview. Images are indexed [y, x].

    points_pix_subviews = calcam_calib.project_points(points_xyz, fill_value=None, coords=image_coords)
    points_pix, info = select_visible_points_from_subviews(points_pix_subviews, subview_mask, points_xyz=points_xyz,
                                    subviews_keep='all',
                                    raise_on_duplicate_view=True, raise_on_out_of_frame=raise_on_out_of_frame)

    return points_pix, info

def toroidal_to_pixel_coordinates(calcam_calib, points_rzphi, angle_units='degrees', image_coords='Display',
                                  raise_on_out_of_frame=True):
    """Project toroidal spatial coordinates onto the camera field of view, properly handling sub-views properly

    Args:
        calcam_calib: Calcam calibration object
        points_rzphi: Array (npoints x 3) of toroidal spatial coordiantes ordered: [[r, z, phi], [...], ...]
        angle_units: Units for phi (radians/degrees)
        image_coords: Project to calcam coords Display/Original

    Returns: Array (npoints, 2) of pixel coordiantes, dict of subview visibility info

    """
    points_rzphi = np.array(points_rzphi)
    if points_rzphi.ndim == 1:
        points_rzphi = points_rzphi[np.newaxis, :]
    r, z, phi = points_rzphi[:, 0], points_rzphi[:, 1], points_rzphi[:, 2]

    x, y = cylindrical_to_cartesian(r, phi, angles_units=angle_units)
    points_xyz = np.array([x, y, z]).T

    points_pix, info = cartesian_to_pixel_coordinates(calcam_calib, points_xyz, image_coords=image_coords,
                                                      raise_on_out_of_frame=raise_on_out_of_frame)

    return points_pix, info

def select_visible_points_from_subviews(points_pix_subviews, subview_mask, points_xyz=None, subviews_keep='all',
                                        raise_on_duplicate_view=True, raise_on_out_of_frame=True, subview_default=None):
    """Select projected points that are visible in that subview.

    When Calcam projects (x,y,z) coordinates onto an image, it returns are list of arrays of projected points for
    each subview. Typically given (x,y,z) points will only be visible in one subview, so this function can be used to
    pick out the projected points that are visible in each subview according to the subview_mask.

    Args:
        points_pix_subviews: List of Nx2 NumPY arrays containing the image coordinates returned by project_points
        subview_mask: Integer image mask indicating which subview each pixel belongs to
        points_xyz: (Optional) Cartesian coordinates of projected points. If provided, also return filtered input coords
        subviews_keep: Whether to only select points from a particular subview (default='all')
        raise_on_duplicate_view: Raise exception if a point is imaged in multiple subviews
        raise_on_out_of_frame: Raise exception if a projected point is not in frame
        subview_default: Subview index to use for returning our of frame pixel coords (default 0 for single subview)
                         (ie if raise_on_out_of_frame=False)

    Returns: (points, info) where poinits is a Nx2 NumPY array of projected points that fall within their respective
    subview's masked area and info is a dict of additional info

    """
    if subviews_keep == 'all':
        subviews_keep = list(set(subview_mask.flatten()))
    if (not raise_on_out_of_frame) and (len(points_pix_subviews) == 1) and (subview_default is None):
        # Only one subview, so safe to return out of frame pixels for subview 0 (otherwise ambiguous)
        subview_default = 0

    # Calcam uses convention that the origin (0,0) is in the centre of the top-left pixel - reverse for indexing
    # subview_mask = subview_mask[::-1, :]

    info = {'n_subviews': len(points_pix_subviews)}
    info['mask_in_frame'] = defaultdict(dict)
    info['mask_in_subview'] = defaultdict(dict)
    info['points_visible'] = defaultdict(dict)
    points_keep = []
    subview_origin = []
    # TODO: If in future paths need to span sub-views, switch to looping over input points to preserve path order
    for i_subview, points in enumerate(points_pix_subviews):
        if i_subview not in subviews_keep:
            continue
        # Switch order of x and y coords for indexing mask array
        # points_int = tuple(points[:, ::-1].astype(np.intp))
        points_int = points.astype(np.intp)
        mask_in_frame = check_in_frame(points_int[:, 0], points_int[:, 1], subview_mask.shape)

        # Of points inside the frame find which belong to this subview
        mask_in_subview = copy(mask_in_frame)
        mask_values = subview_mask[points_int[mask_in_frame, 1], points_int[mask_in_frame, 0]]
        mask_in_subview[mask_in_frame] = mask_values == i_subview
        points_in_subview = points[mask_in_subview, :]

        info['mask_in_frame'][i_subview] = mask_in_frame
        info['mask_in_subview'][i_subview] = mask_in_subview
        info['points_visible'][i_subview] = points_in_subview

        if raise_on_out_of_frame:
            # Only keep points in frame
            points_keep.append(points_in_subview)
            subview_origin.append(np.full(len(points_in_subview), i_subview))
        else:
            if (i_subview == subview_default):
                # Return out of frame points from default subview
                # TODO: In case of multiple sub-views, avoid duplication of points
                points_keep.append(points)
                subview_origin.append(np.full(len(points), i_subview))
            else:
                points_keep.append(points_in_subview)
                subview_origin.append(np.full(len(points_in_subview), i_subview))

    points_keep = np.concatenate(points_keep)
    subview_origin = np.concatenate(subview_origin)

    # Check for degenerate projected points that are imaged in multiple sub views
    n_imaged = np.sum([v for v in info['mask_in_subview'].values()], axis=0)
    mask_visible = n_imaged > 0
    n_multi_imaged = np.sum(n_imaged > 1)
    n_missed = np.sum(n_imaged == 0)
    info['mask_visible'] = mask_visible
    info['subview_origin'] = subview_origin
    info['n_imaged'] = n_imaged
    info['n_multi_imaged'] = n_multi_imaged
    info['n_missed'] = n_missed

    if points_xyz is not None:
        points_xyz = np.array(points_xyz)
        info['xyz_visible'] = points_xyz[mask_visible, :]
        info['xyz_out_of_frame'] = points_xyz[~mask_visible, :]
        info['xyz_multi_imaged'] = points_xyz[n_imaged > 1, :]
    else:
        for key in ['xyz_visible', 'xyz_out_of_frame', 'xyz_multi_imaged']:
            info[key] = None

    if n_multi_imaged > 0:
        message = f' {n_multi_imaged} projected points are imaged in multiple sub-views'
        if raise_on_duplicate_view:
            raise ValueError(message)
        else:
            logger.warning(message)
    if n_missed > 0:
        message = (f' {n_missed} projected points are not visible in any sub-view (ie out of frame)\n'
                   f'   (x, y, z): {info["xyz_out_of_frame"]} ')
                   # f'(x_pix, y_pix): {points_keep[~info["mask_visible"]]}')
        if raise_on_out_of_frame:
            raise ValueError(message)
        else:
            logger.warning(message)

    return points_keep, info

def get_calcam_cad_obj(model_name, model_variant, check_fire_cad_defaults=True):
    logger.debug(f'Loading CAD model...');
    t0 = time.time()
    # TODO: Add error messages directing user to update Calcam CAD definitions in settings GUI if CAD not found
    # print(dir(calcam))
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