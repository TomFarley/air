# -*- coding: future_fstrings -*-
#!/usr/bin/env python

"""
Primary analysis workflow for MAST-U/JET IR scheduler analysis code.

Created: 10-10-2019
"""
import fire.plotting.spatial_figures

print(f'Scheduler workflow: Importing modules')
import logging
from typing import Union
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import calcam

import fire
from fire import fire_paths, copy_default_user_settings
from fire.interfaces import interfaces, calcam_calibs
from fire import plugins
from fire.camera import field_of_view, camera_shake, nuc, image_processing
from fire.geometry import geometry
from fire.physics import temperature, heat_flux, physics_parameters
from fire.misc import data_quality, data_structures, utils
from fire.plotting import debug_plots, image_figures
heat_flux_module = heat_flux  # Alias to avoid name clashes

# TODO: remove after debugging core dumps etc
import faulthandler
faulthandler.enable()

# TODO: Remove after debugging pandas warnings
import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# print(logger_info(logger))

cwd = Path(__file__).parent
path_figures = (cwd / '../figures/').resolve()
logger.info(f'Scheduler workflow: Figures will be output to: {path_figures}')

def scheduler_workflow(pulse:Union[int, str], camera:str='rir', pass_no:int=0, machine:str='MAST', scheduler:bool=False,
                       equilibrium:bool=False, update_checkpoints:bool=False, debug:dict=None, figures:dict=None,
                       image_coords = 'Display'):
    """Primary analysis workflow for MAST-U/JET IR scheduler analysis.

    :param pulse: Shot/pulse number or string name for synthetic movie data
    :param camera: Name of camera to analyse (unique name of camera or diagnostic code)
    :param pass_no: Scheduler pass number
    :param machine: Tokamak that the data originates from
    :param shceduler: Has code been called by the scheduler?
    :param equilibrium: Produce additional output with scheduler efit as additional dependency
    :return: Error code
    """
    logger.info(f'Starting scheduler workflow run for inputs: {machine}, {camera}, {pulse}, {pass_no}')
    if debug is None:
        debug = {}
    if figures is None:
        figures = {}

    # Set up data structures
    settings, files, image_data, meta_data = data_structures.init_data_structures()
    if not scheduler:
        config = interfaces.json_load(fire_paths['config'])
    else:
        raise NotImplementedError
    settings['config'] = config

    # Load user's default call arguments
    pulse, camera, machine = utils.update_call_args(config['default_params'], pulse, camera, machine)
    interfaces.check_settings_complete(config, machine, camera)

    # Set up/extract useful meta data
    # TODO: Separate camera name and diag_tag?
    diag_tag = camera
    meta_data.update(dict(pulse=pulse, camera=camera, diag_tag=diag_tag, machine=machine,
                          pass_no=pass_no, status=1, fire_path=fire_paths['root']))
    kwarg_aliases = {'pulse': ['shot'], 'pass_no': ['pass', 'pass_number'], 'fire_path': ['fire']}
    meta_data['kwarg_aliases'] = kwarg_aliases
    device_details = config['machines'][machine]['cameras'][diag_tag]['device_details']
    meta_data.update(device_details)
    # Generate id_strings
    meta_data['id_strings'] = interfaces.generate_pulse_id_strings({}, pulse, camera, machine, pass_no)
    meta_data['variables'] = config['variable_meta_data']
    meta_data.update(config['machines'][machine]['cameras'][diag_tag]['device_details'])

    # Load machine plugins
    machine_plugin_paths = config['paths_input']['plugins']['machine']
    machine_plugin_attrs = config['plugins']['machine']['module_attributes']
    machine_plugins, machine_plugins_info = plugins.plugins.get_compatible_plugins(machine_plugin_paths,
                                                                   attributes_required=machine_plugin_attrs['required'],
                                                                   attributes_optional=machine_plugin_attrs['optional'],
                                                                   plugins_required=machine, plugin_type='machine')
    machine_plugins, machine_plugins_info = machine_plugins[machine], machine_plugins_info[machine]
    fire.active_machine_plugin = (machine_plugins, machine_plugins_info)

    # Load movie plugins compatible with camera
    # TODO: Move movie reader config information to separate json config file
    movie_plugin_paths = config['paths_input']['plugins']['movie']
    movie_plugin_attrs = config['plugins']['movie']['module_attributes']
    movie_plugins_compatible = config['machines'][machine]['cameras'][camera]['movie_plugins']
    # movie_plugins, movie_plugins_info = plugins.plugins.get_compatible_plugins(movie_plugin_paths,
    #                                                        movie_plugin_attrs['required'],
    #                                                        attributes_optional=movie_plugin_attrs['optional'],
    #                                                        plugin_filter=movie_plugins_compatible, plugin_type='movie')

    # Load output format plugins for writing output data
    output_plugin_paths = config['paths_input']['plugins']['output_format']
    output_plugin_attrs = config['plugins']['output_format']['module_attributes']
    output_plugins_active = config['outputs']['active_formats']
    output_variables = config['outputs']['variables']
    output_plugins, output_plugins_info = plugins.plugins.get_compatible_plugins(output_plugin_paths,
                                                                 attributes_required=output_plugin_attrs['required'],
                                                                 attributes_optional=output_plugin_attrs['optional'],
                                                                 plugin_filter=output_plugins_active, plugin_type='output_format')

    # Load movie meta data to get lens and integration time etc.
    movie_paths = config['paths_input']['movie_files']
    movie_fns = config['filenames_input']['movie_files']
    movie_reader = plugins.plugins_movie.MovieReader(movie_plugin_paths, plugin_filter=movie_plugins_compatible,
                                                     movie_paths=movie_paths, movie_fns=movie_fns,
                                                     plugin_precedence=None, movie_plugin_definition_file=None
                                                     )
    movie_meta, movie_origin = movie_reader.read_movie_meta_data(pulse=pulse, camera=camera, machine=machine,
                                                         check_output=True, substitute_unknown_values=(not scheduler))

    # movie_meta, movie_origin = plugins.plugins_movie.read_movie_meta_data(pulse, camera, machine, movie_plugins,
    #                                                 movie_paths=movie_paths, movie_fns=movie_fns,
    #                                                 substitute_unknown_values=(not scheduler))

    meta_data.update(movie_meta)
    # TODO: format trings to numbers eg lens exposure etc
    meta_data.update(field_of_view.calc_field_of_view(meta_data['lens'], pixel_pitch=meta_data['pixel_pitch'],
                                        image_shape=meta_data['detector_resolution']))
    # TODO: Tidy up duplicate names for image_resolution

    # Identify and check existence of input files
    paths_input = config['paths_input']['input_files']
    paths_output = config['paths_output']
    fn_patterns_input = config['filenames_input']
    fn_pattern_checkpoints = config['checkpoints']['filenames']
    params = {'fire_path': fire_paths['root'], 'image_coords': image_coords, **movie_meta}
    files, lookup_info = interfaces.identify_files(pulse, camera, machine, params=params,
                                        search_paths_inputs=paths_input, fn_patterns_inputs=fn_patterns_input,
                                        paths_output=paths_output, fn_pattern_checkpoints=fn_pattern_checkpoints)
    # TODO: Load camera state
    # TODO: Add camera port location to calcam lookup file?
    # TODO: Lookup camera lens, wavelength filter and neutral density filter from separate lookup file?
    camera_settings = utils.convert_dict_values_to_python_types(lookup_info['camera_settings'].to_dict(),
                                                          allow_strings=True, list_delimiters=' ')
    settings['camera_settings'] = camera_settings

    # Load calcam spatial camera calibration
    calcam_calib = calcam.Calibration(load_filename=str(files['calcam_calib']))
    calcam_calib_image_full_frame = calcam_calib.get_image(coords=image_coords)  # Before detector_window applied

    fire.active_calcam_calib = calcam_calib
    meta_data['calcam_calib'] = calcam_calib
    meta_data['calcam_CAD'] = None

    # TODO: Print out summary of analysis settings prior to analysis

    # TODO: Validate frame range etc

    # Load raw frame data
    movie_data, movie_origin = movie_reader.read_movie_data(pulse=pulse, camera=camera, machine=machine,
                                                            n_start=None, n_end=None, stride=1, transforms=None)
    (frame_nos, frame_times, frame_data) = movie_data

    # Use origin information from reading movie meta data to read frame data from same data source (speed & consistency)
    # movie_path = [movie_origin.get('path', None)]
    # movie_fn = [movie_origin.get('fn', None)]
    # movie_plugin = {movie_origin['plugin']: movie_plugins[movie_origin['plugin']]}
    # (frame_nos, frame_times, frame_data), movie_origin = plugins.plugins_movie.read_movie_data(pulse, camera, machine,
    #                                                             movie_plugin,movie_paths=movie_path, movie_fns=movie_fn,
    #                                                             verbose=True)

    #  window = (Left,Top,Width,Height)
    detector_window = movie_meta['detector_window']
    detector_window_info = calcam_calibs.update_detector_window(calcam_calib, frame_data=frame_data,
                                                                detector_window=detector_window, coords='Original')
    calcam_calib_image_windowed = calcam_calib.get_image(coords=image_coords)  # Before detector_window applied

    # Apply transformations (rotate, flip etc.) to get images "right way up" if requested.
    # Must be applied after detector_window
    frame_data = calcam_calibs.apply_frame_display_transformations(frame_data, calcam_calib, image_coords)

    frame_data = utils.movie_data_to_dataarray(frame_data, frame_times, frame_nos, meta_data=meta_data['variables'])



    # Detect saturated pixels, uniform frames etc
    bad_frames_info = data_quality.identify_bad_frames(frame_data, bit_depth=movie_meta['bit_depth'],
                                                       tol_discontinuities=0.01, raise_on_saturated=False)
    frames_nos_discontinuous = bad_frames_info['discontinuous']['frames']['n']
    frame_data, removed_frames = data_quality.remove_bad_opening_and_closing_frames(frame_data,
                                                                                    frames_nos_discontinuous)
    # Now frame_data is in its final form (bad frames removed) merge it into image_data
    image_data = xr.merge([image_data, frame_data])

    # TODO: Lookup and apply t_offset correction to frame times?

    # Fix camera shake
    # TODO: Consider using alternative to first frame for reference, as want bright clear frame with NUC shutter
    # TODO: Consider checking camera rotation which is issue on ASDEX-U
    n_shake_ref = np.percentile(frame_data.n, 85, interpolation='nearest')  # Frame towards video end
    # n_shake_ref = frame_data.mean(dim=('x_pix', 'y_pix')).argmax(dim='n')  # brightest frame (may have shake?)
    frame_shake_ref = frame_data[n_shake_ref]
    # frame_shake_ref = calcam_calib_image_windowed
    erroneous_displacement = 100
    pixel_displacemnts, shake_stats = camera_shake.calc_camera_shake_displacements(frame_data, frame_shake_ref,
                                                        erroneous_displacement=erroneous_displacement, verbose=True)
    pixel_displacemnts = xr.DataArray(pixel_displacemnts, coords={'n': image_data['n'], 'pixel_coord': ['x', 'y']},
                                      dims=['n', 'pixel_coord'])
    image_data['pixel_displacements'] = pixel_displacemnts
    image_data['frame_data'] = camera_shake.remove_camera_shake(frame_data, pixel_displacements=pixel_displacemnts,
                                                        verbose=True)

    if debug.get('camera_shake', False):
        debug_plots.debug_camera_shake(pixel_displacements=pixel_displacemnts)

    # TODO: Pass bad_frames_info to debug plots

    # TODO: Lookup known anommalies to mask - eg stray LP cable, jammed shutter

    # TODO: Detect anommalies
    # TODO: Monitor number of bad pixels for detector health - option to separate odd/even frames for FLIR sensors
    # bad_pixels, frame_no_dead = find_outlier_pixels(frame_raw, tol=30, check_edges=True)
    # TODO: Detect 'twinkling pixels' by looking for pixels with abnormally large ptp variation for steady state images
    # TODO: Detect missing frames in time stamps, regularity of frame rate
    # TODO: Compare sensor temperature change during/between pulses to monitor sensor health, predict frame losses etc.

    # TODO: Rescale DLs to account for window transmission - Move here out of dl_to_temerature()?

    # Apply NUC correction
    # nuc_frame = get_nuc_frame(origin='first_frame', frame_data=frame_data)
    # TODO: Consider using min over whole (good frame) movie range to avoid negative values from nuc subtraction
    # TODO: consider using custon nuc time/frame number range fro each movie? IDL sched had NUC time option
    nuc_frame = nuc.get_nuc_frame(origin={'n': [0, 0]}, frame_data=frame_data, reduce_func='mean')  # Old air sched code used 3rd frame?
    # nuc_frame = get_nuc_frame(origin={'n': [None, None]}, frame_data=frame_data, reduce_func='min')
    frame_data_nuc = nuc.apply_nuc_correction(frame_data, nuc_frame, raise_on_negatives=False)
    image_data['nuc_frame'] = (('y_pix', 'x_pix'), nuc_frame)
    image_data['frame_data_nuc'] = frame_data_nuc

    if debug.get('debug_detector_window', False):  # Need to plot before detector window applied to calibration
        debug_plots.debug_detector_window(detector_window=detector_window, frame_data=image_data,
                                          calcam_calib=calcam_calib, image_full_frame=calcam_calib_image_full_frame,
                                          image_coords=image_coords)
    if debug.get('movie_data_annimation', False):
        image_figures.animate_frame_data(image_data, key='frame_data', nth_frame=1)  #,
                                         # save_path_fn=f'./figures/{machine}-{pulse}-{# camera}-frame_data.gif')
    if debug.get('movie_data_nuc_annimation', False):
        image_figures.animate_frame_data(image_data, key='frame_data_nuc')

    if debug.get('movie_data_nuc', False):
        debug_plots.debug_movie_data(image_data)


    # Get calcam raycast
    raycast_checkpoint_path_fn = files['raycast_checkpoint']
    if raycast_checkpoint_path_fn.exists() and (not update_checkpoints):
        logger.info(f'Reusing existing raycast checkpoint file for image coordiante mapping. '
                    f'Set keyword update_checkpoints=True to recalculate the file.')
        # TODO: Consider saving raydata with calcam.RayData.save() and calculating dataarrays each time
        # Open pre-calculated raycast data to save time
        data_raycast = xr.open_dataset(raycast_checkpoint_path_fn)
        # TODO: Consider only saving raycast data for full frame views and using calcam.RayData.set_detector_window
        x_im, y_im, z_im = (data_raycast[f'{coord}_im'] for coord in ['x', 'y', 'z'])
    else:
        if raycast_checkpoint_path_fn.exists():
            logger.info(f'Reproduing raycast checkpoint file')
        else:
            logger.info(f'Producing raycast checkpoint file for first time for camera={camera}, pulse={pulse}: '
                        f'{raycast_checkpoint_path_fn}')
        # TODO: Make CAD model pulse range dependent
        cad_model_args = config['machines'][machine]['cad_models'][0]
        cad_model = calcam_calibs.get_calcam_cad_obj(**cad_model_args)
        meta_data['calcam_CAD'] = cad_model
        data_raycast = calcam_calibs.get_surface_coords(calcam_calib, cad_model, image_coords=image_coords)
        cad_model.unload()
        x_im, y_im, z_im = (data_raycast[f'{coord}_im'] for coord in ['x', 'y', 'z'])
        # Call machine plugin functions to get s_gloabl, sector, louvre and tile values etc.
        machine_coord_labels = plugins.plugins_machine.get_machine_coordinate_labels(x_im, y_im, z_im,
                                                                       machine_plugins=machine_plugins)
        for key in machine_coord_labels:
            data_raycast[key + '_im'] = (('y_pix', 'x_pix'), machine_coord_labels[key])
        if raycast_checkpoint_path_fn.exists():
            raycast_checkpoint_path_fn.unlink()  # Avoid error overwriting existing file
        data_raycast.to_netcdf(raycast_checkpoint_path_fn)
        logger.info(f'Wrote raycast data to: {raycast_checkpoint_path_fn}')
    image_data = xr.merge([image_data, data_raycast])

    if debug.get('spatial_res', False):
        debug_plots.debug_spatial_res(image_data)

    if figures.get('spatial_res', False):
        fn_spatial_res = path_figures / f'spatial_res_{pulse}_{camera}.png'
        image_figures.figure_spatial_res_max(image_data, clip_range=[None, 20], save_fn=fn_spatial_res, show=True)

    # TODO: call plugin function to get s coordinate along tiles?
    # s_im = get_s_coord_global(x_im, y_im, z_im, machine_plugins)
    # data['s_im'] = (('y_pix', 'x_pix'), s_im)

    if debug.get('spatial_coords', False):
        phi_offset = 112.3
        # phi_offset = 0
        # points_rzphi = [0.8695, -1.33535, 90]  # TS nose hole: R, z, phi
        # points_rzphi = [[0.825, -1.79, 27]]  # R, z, phi
        # points_rzphi = [[0.80, -1.825, phi_offset-42.5], [1.55, -1.825, phi_offset-40]]  # R, z, phi - old IDL
        points_rzphi = None  # R, z, phi
        # analysis path
        # points_rzphi = [[0.80, -1.825, -42.5]]  # R, z, phi - old IDL analysis path
        # points_pix = np.array([[256-137, 320-129], [256-143, 320-0]]) -1.5  # x_pix, y_pix - old IDL analysis path
        points_pix = np.array([[256-133, 320-256], [256-150, 320-41]]) -1.5  # x_pix, y_pix - old IDL analysis path: [256, 133], [41, 150]
        # points_pix = None

        debug_plots.debug_spatial_coords(image_data, points_rzphi=points_rzphi, points_pix=points_pix)

    # TODO: Check if saturated/bad pixels occur along analysis path - update quality flag

    # TODO: Segment/mask image if contains sub-views
    subview_mask = calcam_calib.get_subview_mask(coords='Display')
    subview_indices = set(list(subview_mask.flatten()))
    subview_names = calcam_calib.subview_names

    logger.info(f'Camera calibration contains {calcam_calib.n_subviews} sub-views: {calcam_calib.subview_names}')

    pass

    # TODO: Set sensor subwindow if using full sensor calcam calibration for windowed view
    # calcam_calib.set_detector_window(window=(Left,Top,Width,Height))

    # Identify material surfaces in view
    surface_coords = interfaces.read_csv(files['structure_coords'], sep=', ', index_col='structure')
    r_im, phi_im, z_im = image_data['R_im'], image_data['phi_im'], image_data['z_im']
    visible_structures = geometry.identify_visible_structures(r_im, phi_im, z_im, surface_coords, phi_in_deg=False)
    surface_ids, material_ids, visible_structures, visible_materials = visible_structures

    image_data['surface_id'] = (('y_pix', 'x_pix'), surface_ids)
    image_data['material_id'] = (('y_pix', 'x_pix'), material_ids)
    image_data.attrs['visible_surfaces'] = visible_structures
    image_data.attrs['visible_materials'] = visible_materials

    if debug.get('surfaces', False):
        debug_plots.debug_surfaces(image_data)

    # Read thermal properties of materials for structures in view
    material_names = list(set(visible_materials.values()))
    material_properties = interfaces.json_load(files['material_props'], key_paths=material_names, lists_to_arrays=True)
    image_data.attrs['material_properties'] = material_properties
    # TODO: Segment path according to changes in tile properties

    # TODO: Read temp_bg from file
    temp_bg = 23
    calib_coefs = lookup_info['temperature_coefs'].to_dict()
    # calib_coefs = lookup_pulse_row_in_csv(files['calib_coefs'], pulse=pulse, header=4)
    # TODO: Switch to plugin for temperature calculation if methods on different machines can't be standardised?
    if False and (machine == 'mast'):  # pulse < 50000:
        # TODO: Remove legacy temperature method with old MAST photon count lookup table
        bb_curve = interfaces.read_csv(files['black_body_curve'], index_col='temperature_celcius')
        image_data['temperature_im'] = temperature.dl_to_temerature_legacy(frame_data_nuc, calib_coefs, bb_curve,
                                                           exposure=movie_meta['exposure'],
                                                           solid_angle_pixel=meta_data['solid_angle_pixel'],
                                                           temperature_bg_nuc=temp_bg, meta_data=meta_data['variables'])
    else:
        image_data['temperature_im'] = temperature.dl_to_temerature(frame_data_nuc, calib_coefs, bb_curve=None,
                                                        wavelength_range=camera_settings['wavelength_range'],
                                                        integration_time=movie_meta['exposure'],
                                                        solid_angle_pixel=2*np.pi,  # meta_data['solid_angle_pixel'],
                                                        temperature_bg_nuc=temp_bg, meta_data=meta_data['variables'])

    # TODO: Identify hotspots: MOI 3.2: Machine protection peak tile surface T from IR is 1300 C
    # (bulk tile 250C from T/C)

    if debug.get('temperature_im', False):
        debug_plots.debug_temperature_image(image_data)

    # TODO: Calculate toroidally averaged radial profiles taking into account viewing geometry
    # - may be more complicated than effectively rotating image slightly as in MAST (see data in Thornton2015)

    # TODO: Temporal smoothing of temperature

    # Get spatial coordinates of analysis paths
    # TODO: get material, structure, bad pixel ids along analysis path
    analysis_path_dfn_points_all = interfaces.json_load(files['analysis_path_dfns'])

    # Get names of analysis paths to use in this analysis
    analysis_path_names = lookup_info['analysis_paths']['analysis_path_name'].split(';')
    analysis_path_names = {f'path{i}': name.strip() for i, name in enumerate(analysis_path_names)}
    analysis_path_keys = list(analysis_path_names.keys())

    # TODO: If required project spatial analysis paths into image analysis path defninstions here (ie make spatial dfn
    # optional)

    path_data_all = xr.Dataset()
    for analysis_path_key, analysis_path_name in analysis_path_names.items():
        logger.info(f'Performing analysis along analysis path "{analysis_path_name}" ("{analysis_path_key}")')

        path_coord = f'i_{analysis_path_key}'
        analysis_path_dfn_points = analysis_path_dfn_points_all[analysis_path_name]['coords']
        analysis_path_description = analysis_path_dfn_points_all[analysis_path_name]['description']

        # Get interpolated pixel and spatial coords along path
        frame_masks = {'surface_id': surface_ids, 'material_id': material_ids}
        # TODO: Separate projection and pixel interpolation
        path_data = calcam_calibs.project_spatial_analysis_path(data_raycast, analysis_path_dfn_points, calcam_calib,
                                                                analysis_path_key, masks=frame_masks, image_coords=image_coords)
        # image_data = xr.merge([image_data, path_data])
        path_data_extracted = image_processing.extract_path_data_from_images(image_data, path_data,
                                                                             path_name=analysis_path_key)
        path_data = xr.merge([path_data, path_data_extracted])

        # x_path, y_path, z_path = (path_data[f'{coord}_path'] for coord in ['x', 'y', 'z'])
        # s_path = get_s_coord_path(x_path, y_path, z_path, machine_plugins)
        # path_data['s_path'] = s_path

        if debug.get('poloidal_cross_sec', False):
            fire.plotting.spatial_figures.figure_poloidal_cross_section(image_data=image_data, path_data=path_data, pulse=pulse, no_cal=True,
                                                                        show=True)

        if debug.get('analysis_path', False):
            # TODO: Finish  debug_analysis_path_2d
            # debug_plots.debug_analysis_path_2d(image_data, path_data, path_names=analysis_path_key,
            #                        image_data_in_cross_sections=True, machine_plugins=machine_plugins)
            debug_plots.debug_analysis_path_1d(image_data, path_data, path_names=analysis_path_key,
                                               image_data_in_cross_sections=True, machine_plugins=machine_plugins)
            debug_plots.debug_spatial_coords(image_data, path_data=path_data, path_name=analysis_path_key)

        if debug.get('temperature_path', True):
            debug_plots.debug_temperature_profile_2d(path_data)

        # TODO: Calculate heat fluxes
        heat_flux, extra_results = heat_flux_module.calc_heatflux(image_data['t'], image_data['temperature_im'],
                                                path_data, analysis_path_key, material_properties, visible_materials)
        # Move meta data assignment to function
        heat_flux_key = f'heat_flux_{analysis_path_key}'
        path_data[heat_flux_key] = ((path_coord, 't'), heat_flux)
        path_data[heat_flux_key].attrs.update(meta_data['variables'].get('heat_flux', {}))
        if 'description' in path_data[heat_flux_key].attrs:
            path_data[heat_flux_key].attrs['label'] = path_data[heat_flux_key].attrs['description']
        path_data.coords['t'] = image_data.coords['t']
        path_data = path_data.swap_dims({'n': 't'})

        # TODO: Calculate moving time average and std heat flux profiles against which transients on different time
        # scales can be identified?

        # TODO: Identify peaks due to tile gaps/hot spots that are present at all times - quantify severity?


        # TODO: Calculate physics parameters
        path_data = physics_parameters.calc_physics_params(path_data, analysis_path_key,
                                                           meta_data=meta_data['variables'])
        # TODO: Calculate poloidal target strike angle - important for neutral closure
        # TODO: Account for shadowed area of tiles giving larger thermal mass than assumed - negative q at end on ASDEX-U


        # TODO: Additional calculations with magnetics information for each pixel:
        # midplane coords, connection length, flux expansion (area ratio), target pitch angle

        path_data_all = xr.merge([path_data_all, path_data])


    # TODO: to be compatible with xpad the coords have to be increasing hence if sign is negative we have to reverse the
    # zcoord and reverse the tprofile and qprofile
    # path_fn_out = files['processed_ir_netcdf']

    # TODO: Separate output data selection from ouput code - pass in single dataset, all of which should be written
    # TODO: Define standardised interface for outputs
    # TODO: Pass absolute path to output file?
    output_header_info = config['outputs']['file_header']
    outputs = plugins.plugins_output_format.write_to_output_format(output_plugins, path_data=path_data_all,
                                                                   image_data=image_data,
                           path_names=analysis_path_keys, variable_names_path=output_variables['analysis_path'],
                           variable_names_time=output_variables['time'], variable_names_image=output_variables['image'],
                           device_info=device_details, header_info=output_header_info, meta_data=meta_data)
    # write_processed_ir_to_netcdf(data, path_fn_out)

    logger.info(f'Finished scheduler workflow')

    status = 'success'
    return {'status': status, 'outputs': outputs, 'meta_data': meta_data}


def run_jet():  # pragma: no cover
    pulse = 94935  # Split view example 715285
    camera = 'kldt'
    pass_no = 0
    machine = 'JET'
    scheduler = False
    magnetics = False
    update_checkpoints = False
    # update_checkpoints = True
    debug = {'debug_detector_window': True, 'camera_shake': True,
             'movie_data_annimation': False, 'movie_data_nuc_annimation': False,
             'spatial_coords': False, 'spatial_res': False, 'movie_data_nuc': True,
             'temperature_im': False, 'surfaces': False, 'analysis_path': True}
    # debug = {k: True for k in debug}
    # debug = {k: False for k in debug}
    figures = {'spatial_res': False}
    logger.info(f'Running JET scheduler workflow...')
    status = scheduler_workflow(pulse=pulse, camera=camera, pass_no=pass_no, machine=machine, scheduler=scheduler,
                       equilibrium=magnetics, update_checkpoints=update_checkpoints, debug=debug, figures=figures)
    return status

def run_mast_rir():  # pragma: no cover
    # pulse = 23586  # Full frame with clear spatial calibration
    pulse = 26505  # Full frame OSP only louvre12d, 1D analysis profile, HIGH current - REQUIRES NEW CALCAM CALIBRATION
    # pulse = 26489  # Full frame OSP only, 1D analysis profile, MODERATE current - REQUIRES NEW CALCAM CALIBRATION
    # pulse = 28866  # Low power, (8x320)
    # pulse = 29210  # High power, (8x320) - Lots of bad frames/missing data?
    # pulse = 29936  # Full frame, has good calcam calib
    # pulse = 30378  # High ELM surface temperatures ~450 C

    # pulse = 24688  # full frame - requires new callibration - looking at inner strike point only
    # pulse = 26098  # full frame - no?
    # pulse = 30012  # full frame
    # pulse = 29945  # [  0  80 320  88] TODO: Further check detector window aligned correctly?

    # Pulses with bad detector window meta data (512 widths?): 23775, 26935

    # # pulse_range_rand = [23586, 28840]
    # # pulse_range_rand = [28840, 29936]
    # pulse_range_rand = [29936, 30471]
    # pulse = int(pulse_range_rand[0] + np.diff(pulse_range_rand)[0] * np.random.rand())

    camera = 'rir'
    pass_no = 0
    machine = 'MAST'
    scheduler = False
    magnetics = False
    update_checkpoints = False
    # update_checkpoints = True
    debug = {'debug_detector_window': False, 'camera_shake': False,
             'movie_data_annimation': False, 'movie_data_nuc_annimation': False,
             'spatial_coords': True, 'spatial_res': False, 'movie_data_nuc': False,
             'surfaces': False, 'analysis_path': False, 'temperature_im': False}
    # debug = {k: True for k in debug}
    # debug = {k: False for k in debug}
    figures = {'spatial_res': False}
    logger.info(f'Running MAST scheduler workflow...')
    status = scheduler_workflow(pulse=pulse, camera=camera, pass_no=pass_no, machine=machine, scheduler=scheduler,
                       equilibrium=magnetics, update_checkpoints=update_checkpoints, debug=debug, figures=figures)
    return status

def run_mast_rit():  # pragma: no cover
    # pulse = 30378
    # pulse = 29936
    # pulse = 27880  # Full frame upper divertor
    pulse = 29000  # 80x256 upper divertor
    camera = 'rit'

    pass_no = 0
    machine = 'MAST'
    scheduler = False
    magnetics = False
    update_checkpoints = False
    # update_checkpoints = True
    debug = {'debug_detector_window': True, 'camera_shake': True,
             'movie_data_annimation': False, 'movie_data_nuc_annimation': False,
             'spatial_coords': True, 'spatial_res': False, 'movie_data_nuc': False,
             'surfaces': False, 'analysis_path': True, 'temperature_im': False,}
    # debug = {k: True for k in debug}
    # debug = {k: False for k in debug}
    figures = {'spatial_res': False}
    logger.info(f'Running MAST scheduler workflow...')
    status = scheduler_workflow(pulse=pulse, camera=camera, pass_no=pass_no, machine=machine, scheduler=scheduler,
                       equilibrium=magnetics, update_checkpoints=update_checkpoints, debug=debug, figures=figures)
    return status

def run_mastu():  # pragma: no cover
    pulse = 50000  # CAD view calibrated from test installation images - no plasma
    # pulse = 50001  # Test movie consisting of black body cavity calibration images
    camera = 'rir'
    # camera = 'rit'
    pass_no = 0
    machine = 'MAST_U'
    scheduler = False
    magnetics = False
    update_checkpoints = False
    # update_checkpoints = True
    # TODO: Remove redundant movie_data step
    debug = {'debug_detector_window': True, 'movie_data_annimation': True, 'movie_data_nuc_annimation': False,
             'spatial_coords': False, 'spatial_res': False, 'movie_data_nuc': False,
             'temperature_im': False,'surfaces': False, 'analysis_path': True}
    # debug = {k: True for k in debug}
    # debug = {k: False for k in debug}
    figures = {'spatial_res': False}
    logger.info(f'Running MAST-U scheduler workflow...')
    status = scheduler_workflow(pulse=pulse, camera=camera, pass_no=pass_no, machine=machine, scheduler=scheduler,
                       equilibrium=magnetics, update_checkpoints=update_checkpoints, debug=debug, figures=figures)
    return status

if __name__ == '__main__':
    # delete_file('~/.fire_config.json', verbose=True, raise_on_fail=True)
    copy_default_user_settings(replace_existing=True)

    status = run_mast_rir()
    # status = run_mast_rit()
    # status = run_mastu()
    # status = run_jet()

    outputs = status['outputs']
    if ('uda_putdata' in outputs) and (outputs['uda_putdata']['success']):
        path_fn = outputs['uda_putdata']['path_fn']
        if path_fn.is_file():
            path_fn.unlink()
            logger.info(f'Deleted uda output file to avoid clutter since run from main in scheduler_workflow.py: '
                        f'{path_fn}')