# -*- coding: future_fstrings -*-
#!/usr/bin/env python

"""
Primary analysis workflow for MAST-U/JET IR scheduler analysis code.

Created: 10-10-2019
"""
print(f'Scheduler workflow: Importing modules')
import logging
from typing import Union
from pathlib import Path

import xarray as xr

import calcam

from fire import fire_paths
from fire.interfaces.interfaces import (check_settings_complete, identify_files,
                                        json_load, read_csv, lookup_pulse_row_in_csv,
                                        generate_pulse_id_strings)
from fire.interfaces.plugins_movie import read_movie_meta_data, read_movie_data, check_meta_data
from fire.interfaces.plugins import get_compatible_plugins
from fire.interfaces.calcam_calibs import (get_surface_coords, project_analysis_path,
        apply_frame_display_transformations, get_calcam_cad_obj)
from fire.interfaces.plugins_machine import get_machine_coordinate_labels, get_s_coord_global, get_s_coord_path
from fire.camera.camera_shake import calc_camera_shake_displacements, remove_camera_shake
from fire.camera.nuc import get_nuc_frame, apply_nuc_correction
from fire.camera.image_processing import extract_path_data_from_images
from fire.geometry.geometry import identify_visible_structures
from fire.physics.temperature import dl_to_temerature
from fire.physics.heat_flux import calc_heatflux
from fire.physics.physics_parameters import calc_physics_params
from fire.misc.data_structures import init_data_structures
from fire.misc.data_quality import identify_saturated_frames
from fire.misc.utils import update_call_args, movie_data_to_xarray
from fire.plotting import debug_plots, image_figures
from fire.plotting.debug_plots import (debug_spatial_coords, debug_movie_data, debug_spatial_res,
                                       debug_analysis_path, debug_temperature)
from fire.plotting.image_figures import figure_spatial_res_max

# TODO: remove after debugging core dumps etc
import faulthandler

faulthandler.enable()

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

cwd = Path(__file__).parent
path_figures = (cwd / '../tmp').resolve()
print(f'Scheduler workflow: Figures will be output to: {path_figures}')

def scheduler_workflow(pulse:Union[int, str], camera:str='rir', pass_no:int=0, machine:str='MAST', scheduler:bool=False,
                       equilibrium:bool=False, update_checkpoints:bool=False, debug:dict=None, figures:dict=None):
    """Primary analysis workflow for MAST-U/JET IR scheduler analysis.

    :param pulse: Shot/pulse number or string name for synthetic movie data
    :param camera: Name of camera to analyse (unique name of camera or diagnostic code)
    :param pass_no: Scheduler pass number
    :param machine: Tokamak that the data originates from
    :param shceduler: Has code been called by the scheduler?
    :param equilibrium: Produce additional output with scheduler efit as additional dependency
    :return: Error code
    """
    if debug is None:
        debug = {}
    if figures is None:
        figures = {}
    image_coords = 'Display' if debug else 'Original'
    # Set up data structures
    settings, files, image_data, meta_data = init_data_structures()
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
                                            machine_plugin_attrs['required'],
                                            attributes_optional=machine_plugin_attrs['optional'], plugin_type='machine')
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
    check_meta_data(movie_meta)
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

    # TODO: Print out summary of analysis settings prior to analysis

    # TODO: Validate frame range etc

    # Load raw frame data
    # Use origin information from reading movie meta data to read frame data from same data source (speed & consistency)
    movie_plugin = {movie_origin['plugin']: movie_plugins[movie_origin['plugin']]}
    movie_path = [movie_origin.get('path', None)]
    movie_fn = [movie_origin.get('fn', None)]
    frame_nos, frame_times, frame_data, movie_origin = read_movie_data(pulse, camera, machine, movie_plugin,
                                                                       movie_paths=movie_path, movie_fns=movie_fn, verbose=True)

    # Apply transformations (rotate, flip etc.) to get images "right way up" if requested
    frame_data = apply_frame_display_transformations(frame_data, calcam_calib, image_coords)

    frame_data = movie_data_to_xarray(frame_data, frame_times, frame_nos)
    image_data = xr.merge([image_data, frame_data])

    # TODO: Lookup and apply t_offset correction to frame times?

    # Fix camera shake
    # TODO: Consider using alternative to first frame for reference, as want bright clear frame with NUC shutter
    # TODO: Consider checking camera rotation which is issue on ASDEX-U
    pixel_displacemnts, shake_stats = calc_camera_shake_displacements(frame_data, frame_data[0], verbose=True)
    pixel_displacemnts = xr.DataArray(pixel_displacemnts, coords={'n': image_data['n'], 'pixel_coord': ['x', 'y']},
                                      dims=['n', 'pixel_coord'])
    image_data['pixel_displacements'] = pixel_displacemnts
    image_data['frame_data'] = remove_camera_shake(frame_data, pixel_displacements=pixel_displacemnts, verbose=True)

    # Detect saturated pixels
    saturated_frames = identify_saturated_frames(frame_data, bit_depth=movie_meta['bit_depth'], raise_on_saturated=False)

    # TODO: Lookup anommalies

    # TODO: Detect anommalies
    # TODO: Monitor number of bad pixels for detector health - option to separate odd/even frames for FLIR sensors
    # bad_pixels, frame_no_dead = find_outlier_pixels(frame_raw, tol=30, check_edges=True)
    # TODO: Detect 'twinkling pixels' by looking for pixels with abnormally large ptp variation for steady state images
    # TODO: Detect missing frames in time stamps, regularity of frame rate
    # TODO: Compare sensor temperature change during/between pulses to monitor sensor health, predict frame losses etc.

    # TODO: Rescale DLs to account for window transmission - Move here out of dl_to_temerature()?

    # Apply NUC correction
    # nuc_frame = get_nuc_frame(origin='first_frame', frame_data=frame_data)
    nuc_frame = get_nuc_frame(origin={'n': [2, 2]}, frame_data=frame_data)  # Old air sched code uses 3rd frame?
    # nuc_frame = get_nuc_frame(origin={'n': [None, None]}, frame_data=frame_data, reduce_func='min')
    frame_data_nuc = apply_nuc_correction(frame_data, nuc_frame, raise_on_negatives=False)
    image_data['nuc_frame'] = (('y_pix', 'x_pix'), nuc_frame)
    image_data['frame_data_nuc'] = frame_data_nuc

    if debug.get('movie_data_nuc', False):
        debug_movie_data(image_data)

    # Get calcam raycast
    raycast_checkpoint_path_fn = files['raycast_checkpoint']
    if raycast_checkpoint_path_fn.exists() and (not update_checkpoints):
        logger.info(f'Reusing existing raycast checkpoint file for image coordiante mapping. '
                    f'Set keyword update_checkpoints=True to recalculate the file.')
        # Open pre-calculated raycast data to save time
        data_raycast = xr.open_dataset(raycast_checkpoint_path_fn)
        x_im, y_im, z_im = (data_raycast[f'{coord}_im'] for coord in ['x', 'y', 'z'])
    else:
        if raycast_checkpoint_path_fn.exists():
            logger.info(f'Reproduing raycast checkpoint file')
        else:
            logger.info(f'Producing raycast checkpoint file for first time for camera={camera}, pulse={pulse}')
        # TODO: Make CAD model pulse range dependent
        cad_model_args = config['machines'][machine]['cad_models'][0]
        cad_model = get_calcam_cad_obj(**cad_model_args)
        meta_data['calcam_CAD'] = cad_model
        data_raycast = get_surface_coords(calcam_calib, cad_model, image_coords=image_coords)
        cad_model.unload()
        x_im, y_im, z_im = (data_raycast[f'{coord}_im'] for coord in ['x', 'y', 'z'])
        # Call machine plugin functions to get s_gloabl, sector, louvre and tile values etc.
        machine_coord_labels = get_machine_coordinate_labels(x_im, y_im, z_im, machine_plugins=machine_plugins)
        for key in machine_coord_labels:
            data_raycast[key + '_im'] = (('y_pix', 'x_pix'), machine_coord_labels[key])
        if raycast_checkpoint_path_fn.exists():
            raycast_checkpoint_path_fn.unlink()  # Avoid error overwriting existing file
        data_raycast.to_netcdf(raycast_checkpoint_path_fn)
        logger.info(f'Wrote raycast data to: {raycast_checkpoint_path_fn}')
    image_data = xr.merge([image_data, data_raycast])

    if debug.get('spatial_res', False):
        debug_spatial_res(image_data)

    if figures.get('spatial_res', False):
        fn_spatial_res = path_figures / f'spatial_res_{pulse}_{camera}.png'
        figure_spatial_res_max(image_data, save_fn=fn_spatial_res, show=True)

    # TODO: call plugin function to get s coordinate along tiles?
    # s_im = get_s_coord_global(x_im, y_im, z_im, machine_plugins)
    # data['s_im'] = (('y_pix', 'x_pix'), s_im)

    if debug.get('spatial_coords', False):
        debug_spatial_coords(image_data)

    # TODO: Check if saturated/bad pixels occur along analysis path - update quality flag

    # TODO: Segment/mask image if contains sub-views

    # Identify material surfaces in view
    surface_coords = read_csv(files['structure_coords'], sep=', ', index_col='structure')
    r_im, phi_im, z_im = image_data['R_im'], image_data['phi_im'], image_data['z_im']
    surface_ids, material_ids, visible_structures, visible_materials = identify_visible_structures(r_im, phi_im, z_im,
                                                                                                   surface_coords, phi_in_deg=False)
    image_data['surface_id'] = (('y_pix', 'x_pix'), surface_ids)
    image_data['material_id'] = (('y_pix', 'x_pix'), material_ids)
    image_data.attrs['visible_surfaces'] = visible_structures
    image_data.attrs['visible_materials'] = visible_materials

    if debug.get('surfaces', False):
        debug_plots.debug_surfaces(image_data)

    # Read thermal properties of materials for structures in view
    material_names = list(set(visible_materials.values()))
    material_properties = json_load(files['material_props'], key_paths=material_names, lists_to_arrays=True)
    image_data.attrs['material_properties'] = material_properties
    # TODO: Segment path according to changes in tile properties

    # TODO: Read temp_bg from file
    temp_bg = 23
    # TODO: Calculate BB curve analytically
    bb_curve = read_csv(files['black_body_curve'], index_col='temperature_celcius')
    calib_coefs = lookup_info['temperature_coefs'].to_dict()
    # calib_coefs = lookup_pulse_row_in_csv(files['calib_coefs'], pulse=pulse, header=4)
    image_data['temperature_im'] = dl_to_temerature(frame_data_nuc, calib_coefs, bb_curve,
                                                 exposure=movie_meta['exposure'], temp_nuc_bg=temp_bg)

    # TODO: Identify hotspots: MOI 3.2: Machine protection peak tile surface T from IR is 1300 C
    # (bulk tile 250C from T/C)

    if debug.get('temperature', False):
        debug_temperature(image_data)

    # TODO: Calculate toroidally averaged radial profiles taking into account viewing geometry
    # - may be more complicated than effectively rotating image slightly as in MAST (see data in Thornton2015)

    # TODO: Temporal smoothing of temperature

    # Get spatial and pixel coordinates of analysis path
    # TODO: get material, structure, bad pixel ids along analysis path
    analysis_path_dfn_points_all = json_load(files['analysis_path_dfns'])
    frame_masks = {'surface_id': surface_ids, 'material_id': material_ids}
    analysis_path_name_full = lookup_info['analysis_paths']['analysis_path_name']
    # TODO: Cleanly handle multiple analysis paths
    analysis_path_names = {f'path{i}': name for i, name in enumerate([analysis_path_name_full])}
    analysis_path_name_short = list(analysis_path_names.keys())[0]
    path_coord = f'i_{analysis_path_name_short}'
    # TODO: Top level loop over analysis paths?
    analysis_path_dfn_points = analysis_path_dfn_points_all[analysis_path_name_full]

    # Get interpolated pixel and spatial coords along path
    path_data = project_analysis_path(data_raycast, analysis_path_dfn_points, calcam_calib, analysis_path_name_short,
                                      masks=frame_masks)
    # image_data = xr.merge([image_data, path_data])
    path_data_extracted = extract_path_data_from_images(image_data, path_data, path_name=analysis_path_name_short)
    path_data = xr.merge([path_data, path_data_extracted])

    # x_path, y_path, z_path = (path_data[f'{coord}_path'] for coord in ['x', 'y', 'z'])
    # s_path = get_s_coord_path(x_path, y_path, z_path, machine_plugins)
    # path_data['s_path'] = s_path

    if debug.get('poloidal_cross_sec', False):
        image_figures.figure_poloidal_cross_section(image_data=image_data, path_data=path_data, pulse=pulse, no_cal=True,
                                                    show=True)

    if debug.get('analysis_path', False):
        debug_analysis_path(image_data, path_data, path_name=analysis_path_name_short)

    # TODO: Calculate heat fluxes
    heat_flux, extra_results = calc_heatflux(image_data['t'], image_data['temperature_im'], path_data,
                                             analysis_path_name_short, material_properties, visible_materials)
    path_data[f'heat_flux_{analysis_path_name_short}'] = (('t', path_coord), heat_flux)
    path_data.coords['t'] = image_data.coords['t']
    path_data = path_data.swap_dims({'n': 't'})

    # TODO: Calculate moving time average and std heat flux profiles against which transients on different time
    # scales can be identified?

    # TODO: Identify peaks due to tile gaps/hot spots that are present at all times - quantify severity?


    # TODO: Calculate physics parameters
    path_data = calc_physics_params(path_data, analysis_path_name_short)
    # TODO: Calculate poloidal target strike angle - important for neutral closure
    # TODO: Account for shadowed area of tiles giving larger thermal mass than assumed - negative q at end on ASDEX-U


    # TODO: Additional calculations with magnetics information for each pixel:
    # midplane coords, connection length, flux expansion (area ratio), target pitch angle

    # TODO: Write output netcdf file - call machine specific plugin
    # NOTE: to be compatible with xpad the coords haveto increasing hence if sign is negative we have to reverse the
    # zcoord and reverse the tprofile and qprofile
    path_fn_out = files['processed_ir_netcdf']
    # write_processed_ir_to_netcdf(data, path_fn_out)

    print(f'Finished scheduler workflow')
    return 0


def run_mast():  # pragma: no cover
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
    debug = {'movie_data': False, 'spatial_coords': False, 'spatial_res': False, 'movie_data_nuc': False,
             'surfaces': False, 'analysis_path': False, 'temperature': False}
    # debug = {k: True for k in debug}
    figures = {'spatial_res': False}
    print(f'Running MAST scheduler workflow...')
    scheduler_workflow(pulse=pulse, camera=camera, pass_no=pass_no, machine=machine, scheduler=scheduler,
                       equilibrium=magnetics, update_checkpoints=update_checkpoints, debug=debug, figures=figures)

def run_mastu():  # pragma: no cover
    pulse = 50000  # Test installation images - no plasma
    camera = 'rir'
    pass_no = 0
    machine = 'MAST_U'
    scheduler = False
    magnetics = False
    update_checkpoints = False
    # update_checkpoints = True
    debug = {'movie_data': False, 'spatial_coords': False, 'spatial_res': False, 'movie_data_nuc': False,
             'temperature': False, 'surfaces': False, 'analysis_path': True}
    # debug = {k: True for k in debug}
    figures = {'spatial_res': False}
    print(f'Running MAST-U scheduler workflow...')
    scheduler_workflow(pulse=pulse, camera=camera, pass_no=pass_no, machine=machine, scheduler=scheduler,
                       equilibrium=magnetics, update_checkpoints=update_checkpoints, debug=debug, figures=figures)

if __name__ == '__main__':
    # run_mast()
    run_mastu()
