#!/usr/bin/env python

"""
Primary analysis workflow for MAST-U/JET IR scheduler analysis code.

Created: 10-10-2019
"""

from typing import Union, Iterable, Optional
from pathlib import Path

import numpy as np
import xarray as xr

import calcam

from fire.data_structures import init_data_structures
from fire.interfaces.interfaces import (load_user_defaults, identify_input_files, read_movie_meta_data,
    read_movie_data, generate_shot_id_strings)
from fire.utils import update_call_args

def scheduler_workflow(shot:Union[int, str], camera:str='rir', pass_no:int=0, machine:str='MAST', shceduler:bool=False,
                       magnetics:bool=False):
    """Primary analysis workflow for MAST-U/JET IR scheduler analysis.

    :param shot: Shot/pulse number or string name for synthetic movie data
    :param camera: Name of camera to analyse (unique name of camera or diagnostic code)
    :param pass_no: Scheduler pass number
    :param machine: Tokamak that the data originates from
    :param shceduler: Has code been called by the scheduler?
    :param magnetics: Produce additional output with scheduler efit as additional dependency
    :return: Error code
    """
    # Set up data structures
    settings, files, data, meta_data = init_data_structures()

    # Load user's default call arguments
    settings['user_defaults'] = load_user_defaults()
    shot, camera, machine = update_call_args(settings['user_defaults'], shot, camera, machine)

    # Generate id_strings
    meta_data['id_strings'] = generate_shot_id_strings(shot, camera, machine, pass_no)

    # Idenify and check existence of input files
    files = identify_input_files(shot, camera, machine)

    # Load camera state
    # settings['camera_state'] = get_camera_state(shot, camera, machine)

    # Load movie meta data
    meta_data = read_movie_meta_data(shot, camera, machine)

    # Validate frame range etc

    # Load raw frame data
    frame_nos, frame_times, frame_data = read_movie_data(shot, camera, machine)

    # Load calcam spatial camera calibration
    meta_data['calcam_calib'] = calcam.Calibration(load_filename=files['calcam_calib'])

    # Segment/mask image if contains sub-views


    # Detect saturation


    # Lookup anommalies


    # Detect anommalies


    # Fix camera shake


    # Apply NUC correction


    # Segment image according to tiles/material properties


    # Convert raw DL to temperature


    # Load analysis path


    # Calculate heat fluxes


    # Calculate physics parameters


    # Write output file


    return 0


