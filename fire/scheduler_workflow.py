#!/usr/bin/env python

"""
Primary analysis workflow for MAST-U/JET IR scheduler analysis code.

Created: 10-10-2019
"""

from typing import Union, Iterable, Optional
from pathlib import Path

import numpy as np
import xarray as xr

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


    # Idenify and check existence of input files


    # Load raw IR data


    # Load calcam spatial camera calibration


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


