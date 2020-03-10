# -*- coding: future_fstrings -*-
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
from fire.interfaces.interfaces import (check_settings_complete, identify_files,
                                        json_load, read_csv, lookup_pulse_row_in_csv,
                                        generate_pulse_id_strings)
from fire.interfaces.plugins_movie import read_movie_meta_data, read_movie_data
from fire.interfaces.plugins import get_compatible_plugins
from fire.interfaces.calcam_calibs import get_surface_coords, project_analysis_path, apply_frame_display_transformations
from fire.camera_shake import calc_camera_shake_displacements, remove_camera_shake
from fire.geometry import identify_visible_structures
from fire.interfaces.plugins_machine import get_machine_location_labels, get_s_coord_global, get_s_coord_path
from fire.utils import update_call_args, movie_data_to_xarray
from fire.nuc import get_nuc_frame, apply_nuc_correction
from fire.data_quality import identify_saturated_frames
from fire.temperature import dl_to_temerature
from fire.heat_flux import calc_heatflux
from fire.interfaces.ouput_data_plugins.output_netcdf import write_processed_ir_to_netcdf

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def scheduler_workflow(pulse:Union[int, str], camera:str='rir', pass_no:int=0, machine:str='MAST', scheduler:bool=False,
                       magnetics:bool=False, update_checkpoints:bool=False, debug:bool=False):
    """Primary analysis workflow for MAST-U/JET IR scheduler analysis.

    :param pulse: Shot/pulse number or string name for synthetic movie data
    :param camera: Name of camera to analyse (unique name of camera or diagnostic code)
    :param pass_no: Scheduler pass number
    :param machine: Tokamak that the data originates from
    :param shceduler: Has code been called by the scheduler?
    :param magnetics: Produce additional output with scheduler efit as additional dependency
    :return: Error code
    """
    image_coords = 'Display' if debug else 'Original'
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

    # Load machine plugins
    machine_plugin_paths = config['paths_input']['machine_plugins']
    machine_plugin_attrs = config['plugin_attributes']['machine']
    machine_plugins, machine_plugins_info = get_compatible_plugins(machine_plugin_paths,
                                machine_plugin_attrs['required'], attributes_optional=machine_plugin_attrs['optional'],
                                plugin_type='machine')
    machine_plugins, machine_plugins_info = machine_plugins[machine], machine_plugins_info[machine]

    # Load movie plugins compatible with camera
    movie_plugin_paths = config['paths_input']['movie_plugins']
    movie_plugin_attrs = config['plugin_attributes']['movie']
    movie_plugins_compatible = config['machines'][machine]['cameras'][camera]['movie_plugins']
    movie_plugins, movie_plugins_info = get_compatible_plugins(movie_plugin_paths, movie_plugin_attrs['required'],
                                           attributes_optional=movie_plugin_attrs['optional'],
                                           plugin_filter=movie_plugins_compatible, plugin_type='movie')

    # Load movie meta data to get lens and integration time etc.
    movie_paths = config['paths_input']['movie_files']
    movie_fns = config['filenames_input']['movie_files']
    movie_meta, movie_origin = read_movie_meta_data(pulse, camera, machine, movie_plugins,
                                     movie_paths=movie_paths, movie_fns=movie_fns)

    # Identify and check existence of input files
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
    calcam_calib = calcam.Calibration(load_filename=str(files['calcam_calib']))
    # TODO: calcam_calib.set_detector_window(window)  #  window = (Left,Top,Width,Height)

    meta_data['calcam_calib'] = calcam_calib
    meta_data['calcam_CAD'] = None

    # TODO: Validate frame range etc

    # Load raw frame data
    frame_nos, frame_times, frame_data, movie_origin = read_movie_data(pulse, camera, machine, movie_plugins,
                                                        movie_paths=movie_paths, movie_fns=movie_fns, verbose=True)

    # Apply transformations (rotate, flip etc.) to get images "right way up" if requested
    frame_data = apply_frame_display_transformations(frame_data, calcam_calib, image_coords)

    frame_data = movie_data_to_xarray(frame_data, frame_times, frame_nos)
    data = xr.merge([data, frame_data])

    # Fix camera shake
    # TODO: Consider using alternative to first frame for reference, as want bright clear frame with NUC shutter
    # TODO: Consider checking camera rotation which is issue on ASDEX-U
    pixel_displacemnts, shake_stats = calc_camera_shake_displacements(frame_data, frame_data[0], verbose=True)
    pixel_displacemnts = xr.DataArray(pixel_displacemnts, coords={'n': data['n'], 'pixel_coord': ['x', 'y']},
                                                                  dims=['n', 'pixel_coord'])
    data['pixel_displacements'] = pixel_displacemnts
    data['frame_data'] = remove_camera_shake(frame_data, pixel_displacements=pixel_displacemnts, verbose=True)

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
    x_im, y_im, z_im = (data_raycast[f'{coord}_im'] for coord in ['x', 'y', 'z'])

    # TODO: call plugin function to get sector, louvre and tile values
    machine_location_labels = get_machine_location_labels(x_im, y_im, z_im, machine_plugins=machine_plugins)
    for key in machine_location_labels:
        data[key+'_im'] = (('y_pix', 'x_pix'), machine_location_labels[key])

    # TODO: call plugin function to get s coordinate along tiles?
    s_im = get_s_coord_global(x_im, y_im, z_im, machine_plugins)
    data['s_im'] = (('y_pix', 'x_pix'), s_im)

    # TODO: Check if saturated/bad pixels occur along analysis path - update quality flag

    # TODO: Segment/mask image if contains sub-views

    # Identify material surfaces in view
    surface_coords = read_csv(files['structure_coords'], sep=', ', index_col='structure')
    r_im, phi_im, z_im = data['R_im'], data['phi_im'], data['z_im']
    surface_ids, material_ids, visible_structures, visible_materials = identify_visible_structures(r_im, phi_im, z_im,
                                                                                     surface_coords, phi_in_deg=False)
    data['surface_id'] = (('y_pix', 'x_pix'), surface_ids)
    data['material_id'] = (('y_pix', 'x_pix'), material_ids)
    data.attrs['visible_surfaces'] = visible_structures
    data.attrs['visible_materials'] = visible_materials

    # Read thermal properties of materials for structures in view
    material_names = list(set(visible_materials.values()))
    material_properties = json_load(files['material_props'], key_paths=material_names, lists_to_arrays=True)
    data.attrs['material_properties'] = material_properties
    # TODO: Segment path according to changes in tile properties

    # Detect saturated pixels
    saturated_frames = identify_saturated_frames(frame_data, bit_depth=movie_meta['bit_depth'], raise_on_saturated=False)

    # TODO: Lookup anommalies

    # TODO: Detect anommalies
    # TODO: Monitor number of bad pixels for detector health - option to separate odd/even frames for FLIR sensors
    # TODO: Detect 'twinkling pixels' by looking for pixels with abnormally large ptp variation for steady state images
    # TODO: Detect missing frames in time stamps, regularity of frame rate
    # TODO: Compare sensor temperature change during/between pulses to monitor sensor health, predict frame losses etc.

    # Get spatial and pixel coordinates of analysis path
    # TODO: get material, structure, bad pixel ids along analysis path
    analysis_path_dfn_points = json_load(files['analysis_path_dfns'])
    frame_masks = {'surface_id': surface_ids, 'material_id': material_ids}
    analysis_path = project_analysis_path(data_raycast, analysis_path_dfn_points, calcam_calib, masks=frame_masks)
    x_path, y_path, z_path = (analysis_path[f'{coord}_path'] for coord in ['x', 'y', 'z'])
    s_path = get_s_coord_path(x_path, y_path, z_path, machine_plugins)
    analysis_path['s_path'] = s_path
    data = xr.merge([data, analysis_path])

    # Apply NUC correction
    # nuc_frame = get_nuc_frame(origin='first_frame', frame_data=frame_data)
    nuc_frame = get_nuc_frame(origin={'n': [2, 2]}, frame_data=frame_data)  # Old air sched code uses 3rd frame?
    # nuc_frame = get_nuc_frame(origin={'n': [None, None]}, frame_data=frame_data, reduce_func='min')
    frame_data_nuc = apply_nuc_correction(frame_data, nuc_frame, raise_on_negatives=False)
    data['frame_data_nuc'] = frame_data_nuc


    # TODO: Read temp_bg from file
    temp_bg = 23
    # TODO: Convert raw DL to temperature
    bb_curve = read_csv(files['black_body_curve'], index_col='temperature_celcius')
    calib_coefs = lookup_pulse_row_in_csv(files['calib_coefs'], pulse=pulse, header=4)
    data['frame_temperature'] = dl_to_temerature(frame_data_nuc, calib_coefs, bb_curve,
                                                 exposure=movie_meta['exposure'], temp_bg=temp_bg)

    # TODO: Calculate toroidally averaged radial profiles taking into account viewing geometry
    # - may be more complicated than effectively rotating image slightly as in MAST (see data in Thornton2015)

    # TODO: Temporal smoothing of temperature


    # TODO: Calculate heat fluxes


    # TODO: Calculate moving time average and std heat flux profiles against which transients on different time
    # scales can be identified?

    # TODO: Identify peaks due to tile gaps/hot spots that are present at all times - quantify severity?


    # TODO: Calculate physics parameters
    # TODO: Calculate poloidal target strike angle - important for neutral closure
    # TODO: Account for shadowed area of tiles leading to larger themal mass than assumed - gives negative q after
    # discharge on ASDEX-U
    data['heat_flux'] = calc_heatflux(data['t'], data['frame_temperature'], analysis_path, material_properties,
                                      visible_materials)

    # TODO: Additional calculations with magnetics information for each pixel:
    # midplane coords, connection length, flux expansion (area ratio), target pitch angle

    # TODO: Write output file - call machine specific plugin
    path_fn_out = files['processed_ir_netcdf']
    # write_processed_ir_to_netcdf(data, path_fn_out)

    print(f'Finished scheduler workflow')
    return 0


def run_mast():
    # pulse = 30378
    # camera = 'rit'
    pulse = 23586  # Full frame with clear spatial calibration
    # pulse = 28866  # Low power, (8x320)
    # pulse = 29210  # High power, (8x320)
    # pulse = 30378  # High ELM surface temperatures ~450 C
    camera = 'rir'
    pass_no = 0
    machine = 'MAST'
    scheduler = False
    magnetics = False
    update_checkpoints = False
    # update_checkpoints = True
    debug = True
    print(f'Running MAST scheduler workflow...')
    scheduler_workflow(pulse=pulse, camera=camera, pass_no=pass_no, machine=machine, scheduler=scheduler,
                       magnetics=magnetics, update_checkpoints=update_checkpoints, debug=debug)

def run_mastu():
    pulse = 50000  # Test installation images - no plasma
    camera = 'rir'
    pass_no = 0
    machine = 'MAST_U'
    scheduler = False
    magnetics = False
    update_checkpoints = False
    # update_checkpoints = True
    debug = True
    print(f'Running MAST-U scheduler workflow...')
    scheduler_workflow(pulse=pulse, camera=camera, pass_no=pass_no, machine=machine, scheduler=scheduler,
                       magnetics=magnetics, update_checkpoints=update_checkpoints, debug=debug)

if __name__ == '__main__':
    # run_mast()
    run_mastu()
