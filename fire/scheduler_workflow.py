#!/usr/bin/env python

"""
Primary analysis workflow for MAST-U/JET IR scheduler analysis code.

Created: 10-10-2019
"""
import logging, time
from typing import Union, Iterable, Optional
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import calcam

from fire import fire_paths
from fire.data_structures import init_data_structures
from fire.interfaces.interfaces import (check_settings_complete, get_compatible_movie_plugins, identify_files,
                                        json_load, read_csv, lookup_pulse_row_in_csv,
                                        read_movie_meta_data, read_movie_data, generate_pulse_id_strings)
from fire.interfaces.calcam_calibs import get_surface_coords, project_analysis_path, apply_frame_display_transformations
from fire.utils import update_call_args, movie_data_to_xarray
from fire.nuc import get_nuc_frame, apply_nuc_correction
from fire.data_quality import identify_saturated_frames
from fire.temperature import dl_to_temerature

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def scheduler_workflow(pulse:Union[int, str], camera:str='rir', pass_no:int=0, machine:str='MAST', scheduler:bool=False,
                       magnetics:bool=False, update_checkpoints=False):
    """Primary analysis workflow for MAST-U/JET IR scheduler analysis.

    :param pulse: Shot/pulse number or string name for synthetic movie data
    :param camera: Name of camera to analyse (unique name of camera or diagnostic code)
    :param pass_no: Scheduler pass number
    :param machine: Tokamak that the data originates from
    :param shceduler: Has code been called by the scheduler?
    :param magnetics: Produce additional output with scheduler efit as additional dependency
    :return: Error code
    """
    image_coords = 'Display'
    # Set up data structures
    settings, files, data, meta_data = init_data_structures()
    if not scheduler:
        config = json_load(fire_paths['config'])
    else:
        raise NotImplementedError
    settings['config'] = config

    # Load user's default call arguments
    pulse, camera, machine = update_call_args(config['default_params'], pulse, camera, machine)
    check_settings_complete(config, machine, camera)
    # Generate id_strings
    meta_data['id_strings'] = generate_pulse_id_strings({}, pulse, camera, machine, pass_no)

    # Load movie meta data to get lens and integration time etc.
    movie_plugins = get_compatible_movie_plugins(config, machine, camera)
    movie_paths = config['paths_input']['movie_files']
    movie_fns = config['filenames_input']['movie_files']
    movie_meta, movie_origin = read_movie_meta_data(pulse, camera, machine, movie_plugins,
                                     movie_paths=movie_paths, movie_fns=movie_fns)

    # Idenify and check existence of input files
    paths_input = config['paths_input']['input_files']
    paths_output = config['paths_output']
    fn_patterns_input = config['filenames_input']
    fn_patterns_output = config['filenames_output']
    params = {'fire_path': fire_paths['root'], 'image_coords': image_coords, **movie_meta}
    files, lookup_info = identify_files(pulse, camera, machine, params=params,
                                        search_paths_inputs=paths_input, fn_patterns_inputs=fn_patterns_input,
                                        paths_output=paths_output, fn_pattern_output=fn_patterns_output)
    # TODO: Load camera state
    # settings['camera_state'] = get_camera_state(pulse, camera, machine)

    # Load calcam spatial camera calibration
    calcam_calib = calcam.Calibration(load_filename=files['calcam_calib'])
    # TODO: calcam_calib.set_detector_window(window)  #  window = (Left,Top,Width,Height)

    meta_data['calcam_calib'] = calcam_calib
    meta_data['calcam_CAD'] = None

    # TODO: Validate frame range etc

    # Load raw frame data
    frame_nos, frame_times, frame_data, movie_origin = read_movie_data(pulse, camera, machine, movie_plugins,
                                                        movie_paths=movie_paths, movie_fns=movie_fns, verbose=True)
    # Apply transformations (rotate, flip etc.) to get images "right way up"
    frame_data = apply_frame_display_transformations(frame_data, calcam_calib, image_coords)

    frame_data = movie_data_to_xarray(frame_data, frame_times, frame_nos)
    data = xr.merge([data, frame_data])

    # Get calcam raycast
    raycast_checkpoint_path_fn = files['raycast_checkpoint']
    if raycast_checkpoint_path_fn.exists() and (not update_checkpoints):
        # Open pre-calculated raycast data to save time
        data_raycast = xr.open_dataset(raycast_checkpoint_path_fn)
    else:
        # TODO: Make CAD model pulse range dependent
        cad_model_args = config['machines'][machine]['cad_models'][0]
        logger.debug(f'Loading CAD model...'); t0 = time.time()
        cad_model = calcam.CADModel(**cad_model_args)
        print(f'Setup CAD model object in {time.time()-t0:1.1f} s')
        meta_data['calcam_CAD'] = cad_model
        data_raycast = get_surface_coords(calcam_calib, cad_model, image_coords=image_coords)
        data_raycast.to_netcdf(raycast_checkpoint_path_fn)
        logger.info(f'Wrote raycast data to: {raycast_checkpoint_path_fn}')
    data = xr.merge([data, data_raycast])

    # Get spatial and pixel coordinates of analysis path
    analysis_path_dfn_points = json_load(files['analysis_path_dfns'])
    analysis_path = project_analysis_path(data_raycast, analysis_path_dfn_points, calcam_calib)
    data = xr.merge([data, analysis_path])

    # TODO: Segment/mask image if contains sub-views

    # Detect saturated pixels
    saturated_frames = identify_saturated_frames(frame_data, bit_depth=movie_meta['bit_depth'], raise_on_saturated=False)

    # TODO: Lookup anommalies

    # TODO: Detect anommalies

    # TODO: Fix camera shake
    # mov = calcam.movement.detect_movement(calcam_calib, moved_im)
    # corrected_image, mask = mov.warp_moved_to_ref(moved_im)
    # updated_calib = calcam.movement.update_calibration(my_calib, moved_im, mov)

    # Apply NUC correction
    # nuc_frame = get_nuc_frame(origin='first_frame', frame_data=frame_data)
    nuc_frame = get_nuc_frame(origin={'n': [2, 2]}, frame_data=frame_data)  # Old air sched code uses 3rd frame
    # nuc_frame = get_nuc_frame(origin={'n': [None, None]}, frame_data=frame_data, reduce_func='min')
    frame_data_nuc = apply_nuc_correction(frame_data, nuc_frame, raise_on_negatives=False)
    data['frame_data_nuc'] = frame_data_nuc
    # TODO: Segment image according to tiles/material properties

    # TODO: Convert raw DL to temperature
    bb_curve = read_csv(files['black_body_curve'], index_col='temperature_celcius')
    calib_coefs = lookup_pulse_row_in_csv(files['calib_coefs'], pulse=pulse, header=4)
    # TODO: Read temp_bg from file
    data['frame_temperature'] = dl_to_temerature(frame_data_nuc, calib_coefs, bb_curve,
                                                 exposure=movie_meta['exposure'], temp_bg=23)

    # TODO: Temporal smoothing of temperature


    # TODO: Calculate heat fluxes


    # TODO: Calculate physics parameters


    # TODO: Write output file

    print(f'Finished scheduler workflow')
    return 0


if __name__ == '__main__':
    # pulse = 30378
    # camera = 'rit'
    pulse = 23586  # Full frame with clear spatial calibration
    # pulse = 28866  # Low power
    # pulse = 29210  # High power
    # pulse = 30378  # High ELM surface temperatures ~450 C
    camera = 'rir'
    pass_no = 0
    machine = 'MAST'
    scheduler = False
    magnetics = False
    update_checkpoints = False
    # update_checkpoints = True
    print(f'Running scheduler workflow...')
    scheduler_workflow(pulse=pulse, camera=camera, pass_no=pass_no, machine=machine, scheduler=scheduler,
                       magnetics=magnetics, update_checkpoints=update_checkpoints)