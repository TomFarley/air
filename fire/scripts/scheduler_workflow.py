# -*- coding: future_fstrings -*-
#!/usr/bin/env python

"""
Primary analysis workflow for MAST-U/JET IR scheduler analysis code.

Created: 10-10-2019
"""
import fire.camera_tools.camera_checks
import fire.plugins.machine_plugins.mast_u

print(f'Scheduler workflow: Importing modules')
import logging
from typing import Union, Optional
from pathlib import Path
from copy import copy

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import calcam

import fire
from fire import fire_paths, copy_default_user_settings
from fire import plugins, camera_tools
from fire.interfaces import interfaces, calcam_calibs, io_basic, io_utils, uda_utils
from fire.camera_tools import field_of_view, camera_shake, nuc, image_processing, camera_checks
from fire.geometry import geometry, s_coordinate
from fire.physics import temperature, heat_flux, physics_parameters
from fire.misc import data_quality, data_structures, utils
from fire.plotting import plot_tools, debug_plots, image_figures, temporal_figures, spatial_figures
heat_flux_module = heat_flux  # Alias to avoid name clashes

# TODO: remove after debugging core dumps etc
import faulthandler
faulthandler.enable()

# TODO: Remove after debugging pandas warnings
import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

# logger = logging.getLogger(__name__)  # TODO: Check propagation of fire_logger file handler
logger = logging.getLogger('fire.scheduler_workflow')  # TODO: Check propagation of fire_logger file handler
# logger.setLevel(logging.DEBUG)
# print(logger_info(logger))

cwd = Path(__file__).parent
# path_figures = (cwd / '../figures/').resolve()
path_figures = (fire_paths['user'] / 'figures/').resolve()
logger.info(f'Scheduler workflow: Figures will be output to: {path_figures}')

def scheduler_workflow(pulse:Union[int, str], camera:str='rir', pass_no:int=0, machine:str='MAST', scheduler:bool=False,
                       equilibrium:bool=False, update_checkpoints:bool=False, analaysis_steps:Optional[list]=None,
                       image_coords = 'Display',
                       debug:Optional[dict]=None, figures:Optional[dict]=None, output:Optional[dict]=None):
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

    if analaysis_steps is None:
        # TODO: Properly set up top level control of analysis steps
        analaysis_steps = dict(camera_shake_correction=False)

    if debug is None:
        debug = {}
    if figures is None:
        figures = {}
    if output is None:
        output = {}

    if machine.lower() == 'mast':
        # nuc_frame_range = [0, 0]
        nuc_frame_range = [2, 2]  # used 3rd frame like IDL sched code?
    else:
        # TODO: Look up frame range before plasma/t=0
        nuc_frame_range = [1, 30]  # used first 30 frames before plasma, ignore first frame due to artifacts

    # TODO: Read temp_bg from file
    temp_bg = 23

    # Set up data structures
    settings, files, image_data, path_data_all, meta_data, meta_runtime = data_structures.init_data_structures()
    meta_data_extra = {}
    if not scheduler:
        config = interfaces.json_load(fire_paths['config'], key_paths_drop=('README',))
    else:
        raise NotImplementedError
    settings['config'] = config

    # Load user's default call arguments
    pulse, camera, machine = utils.update_call_args(config['user']['default_params'], pulse, camera, machine)
    interfaces.check_settings_complete(config, machine, camera)

    # Set up/extract useful meta data
    # TODO: Separate camera name and diag_tag?
    diag_tag_raw = camera
    diag_tag_analysed = uda_utils.get_analysed_diagnostic_tag(diag_tag_raw)
    meta_data.update(dict(pulse=pulse, shot=pulse, camera=camera, diag_tag=diag_tag_raw, diag_tag_raw=diag_tag_raw,
                          diag_tag_analysed=diag_tag_analysed, machine=machine, pass_no=pass_no, status=1, 
                          fire_path=fire_paths['root']))
    kwarg_aliases = {'pulse': ['shot'], 'pass_no': ['pass', 'pass_number'], 'fire_path': ['fire']}
    meta_data['kwarg_aliases'] = kwarg_aliases
    device_details = config['machines'][machine]['cameras'][diag_tag_raw]['device_details']
    meta_data.update(device_details)
    # Generate id_strings
    meta_data['id_strings'] = interfaces.generate_pulse_id_strings({}, pulse, camera, machine, pass_no)
    meta_data['variables'] = config['variable_meta_data']
    meta_data.update(config['machines'][machine]['cameras'][diag_tag_raw]['device_details'])


    # TODO: REMOVE tmp
    import pyuda;
    client = pyuda.Client()
    meta_data_extra['client'] = client
    from fire.interfaces.uda_utils import putdata_create, putdata_close
    file_id, path_fn = putdata_create(fn='ait044852.nc', path='/home/tfarley/repos/air', client=client, **meta_data)
    putdata_close(file_id, client=client)



    # Load machine plugins
    machine_plugin_paths = config['paths_input']['plugins']['machine']
    machine_plugin_attrs = config['plugins']['machine']['module_attributes']
    machine_plugins, machine_plugins_info = plugins.plugins.get_compatible_plugins(machine_plugin_paths,
                                                                   attributes_required=machine_plugin_attrs['required'],
                                                                   attributes_optional=machine_plugin_attrs['optional'],
                                                                   plugins_required=machine, plugin_type='machine')
    machine_plugins, machine_plugins_info = machine_plugins[machine], machine_plugins_info[machine]
    fire.active_machine_plugin = (machine_plugins, machine_plugins_info)
    meta_pulse = plugins.call_plugin_func(machine_plugins, 'pulse_meta_data', args=(pulse,), dummy_ouput={})

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
                                                     plugin_precedence=None, movie_plugin_definition_file=None)
    movie_meta, movie_origin = movie_reader.read_movie_meta_data(pulse=pulse, camera=camera, machine=machine,
                                                         check_output=True, substitute_unknown_values=(not scheduler))

    # movie_meta, movie_origin = plugins.plugins_movie.read_movie_meta_data(pulse, camera, machine, movie_plugins,
    #                                                 movie_paths=movie_paths, movie_fns=movie_fns,
    #                                                 substitute_unknown_values=(not scheduler))

    meta_data.update(movie_meta)
    # TODO: format strings to numbers eg lens exposure etc
    meta_data.update(field_of_view.calc_field_of_view(meta_data['lens'], pixel_pitch=meta_data['pixel_pitch'],
                                        image_shape=meta_data['detector_resolution']))
    # NOTE: meta_data['image_shape'] and ipx header hight/width etc prior to calcam image transformations

    # TODO: Tidy up duplicate names for image_resolution

    # Identify and check existence of input files
    paths_input = config['paths_input']['input_files']
    paths_output = {key: Path(path) for key, path in config['paths_output'].items()}
    fn_patterns_input = config['filenames_input']
    fn_pattern_checkpoints = config['checkpoints']['filenames']
    params = {'fire_path': fire_paths['root'], 'image_coords': image_coords, **movie_meta}
    files, lookup_info = interfaces.identify_files(pulse, camera, machine, params=params,
                                        search_paths_inputs=paths_input, fn_patterns_inputs=fn_patterns_input,
                                        paths_output=paths_output, fn_pattern_checkpoints=fn_pattern_checkpoints)
    meta_data['files'] = files
    logger.info(f'Located input files for analysis: {files}')
    # TODO: Load camera state
    # TODO: Add camera port location to calcam lookup file?
    # TODO: Lookup camera lens, wavelength filter and neutral density filter from separate lookup file?
    camera_settings = utils.convert_dict_values_to_python_types(lookup_info['camera_settings'].to_dict(),
                                                          allow_strings=True, list_delimiters=' ')
    settings['camera_settings'] = camera_settings

    # Load calcam spatial camera calibration
    calcam_calib = calcam.Calibration(load_filename=str(files['calcam_calib']))
    calcam_calib_image_full_frame = calcam_calib.get_image(coords=image_coords)  # Before detector_window applied
    # TODO: Handle different pupil positions for multiple subviews
    subview = 0
    meta_data['calcam_pupilpos'] = calcam_calib.get_pupilpos(subview=subview)
    # TODO: Check that original full frame frame shape matches calcam calibration image shape?

    logger.info(f'Using calcam calibration: {calcam_calib.filename}')

    fire.active_calcam_calib = calcam_calib
    meta_runtime['calcam_calib'] = calcam_calib
    meta_data['calcam_CAD'] = None

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

        # # TODO: remove tmp
        # from fire.interfaces import calcam_cad
        # cell_locator = calcam_cad.get_cell_locator(cad_model)

        # TODO: Add CAD model to separate meta_objects dict to make meta_data easily serialisable
        meta_data_extra['calcam_CAD'] = cad_model
        data_raycast = calcam_calibs.get_surface_coords(calcam_calib, cad_model, image_coords=image_coords,
                                                        remove_long_rays=True)
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
    image_data = xr.merge([image_data, data_raycast])  # , combine_attrs='no_conflicts')


    # TODO: Print out summary of analysis settings prior to analysis

    # TODO: Validate frame range etc

    # Load raw frame data
    movie_data, movie_origin = movie_reader.read_movie_data(pulse=pulse, camera=camera, machine=machine,
                                                            n_start=None, n_end=None, stride=1, transforms=None)
    (frame_nos, frame_times, frame_data) = movie_data

    logger.info("Movie time period [%g, %g]" % (np.nanmin(frame_times), np.nanmax(frame_times)))

    # TODO: Use techniques in generate_calcam_calib_images to identify bright images for plotting elsewhere
    n_middle = int(np.floor(len(frame_data) / 2))  # Frame number in middle of movie
    frame_ref_pre_transforms = copy(frame_data[n_middle])  # Copy of a frame before calcam transformations applied
    calcam_calib_im_pre_transforms = copy(calcam_calib.get_image(coords='Original'))

    #  window = (Left,Top,Width,Height)
    detector_window = movie_meta['detector_window']
    detector_window_info = calcam_calibs.update_detector_window(calcam_calib, frame_data=frame_data,
                                                                detector_window=detector_window, coords='Original')
    calcam_calib_image_windowed = calcam_calib.get_image(coords=image_coords)  # Before detector_window applied

    # Apply transformations (rotate, flip etc.) to get images "right way up" if requested.
    # Must be applied after detector_window
    frame_data = calcam_calibs.apply_frame_display_transformations(frame_data, calcam_calib, image_coords)

    frame_data = data_structures.movie_data_to_dataarray(frame_data, frame_times, frame_nos,
                                                         meta_data=meta_data['variables'])
    image_data = xr.merge([image_data, frame_data])  # , combine_attrs='no_conflicts')

    image_shape = np.array(frame_data.shape[1:])  # NOTE: meta_data['image_shape'] and ipx header info is without image
    # transformations

    if np.any(np.diff(frame_data['t']) < 0):
        logger.warning(f'Movie data contains non-monotonic time data. Re-ordering frames by timestamps')
        frame_data = frame_data.sortby('t')

    if (debug.get('movie_intensity_stats-raw', False)):
        # Force plot if there are any saturated frames
        temporal_figures.plot_movie_intensity_stats(frame_data, meta_data=meta_data, num='movie_intensity_stats-raw')

    if (movie_origin['plugin'] == 'ipx') and (machine == 'mast_u') and (np.max(frame_data) <= 2**12):
        frame_data *= 4  # TODO: Remove when ipx writing fixed!
        logger.warning('Multiplied data from MAST-U ipx file by factor of 4 due to mastvideo ipx writing issue')

    clock_info = fire.plugins.plugins.call_plugin_func(machine_plugins, 'get_camera_external_clock_info',
                                                       args=(camera, pulse))

    # TODO: move time checks to separate function
    if clock_info is not None:
        if (not np.isclose(movie_meta['t_before_pulse'], np.abs(clock_info['clock_t_window'][0]))):
            raise ValueError(f'Movie and camera clock start times do not match. '
                             f'movie={movie_meta["t_before_pulse"]}, clock={clock_info["clock_t_window"]}')
        if np.abs(movie_meta['fps'] - clock_info['clock_frequency']) > 1:
            message = (f'Movie and camera clock frequencies do not match. '
                             f'movie={movie_meta["fps"]}, clock={clock_info["clock_frequency"]}')
            logger.warning(message)
            pass

            if True:
                time_correction_info = fire.plugins.machine_plugins.mast_u.get_frame_time_correction(frame_times,
                                                                                                     clock_info['clock_frame_times'],
                                                                                                     clock_info=clock_info)
                # TODO: Update t_before_pulse to be -ve and rename to 't_movie_start'? - update all meta data files...
                t_offset = np.abs(movie_meta['t_before_pulse'])
                time_correction_factor = (movie_meta['fps']/clock_info['clock_frequency'])
                frame_times = (frame_times + t_offset)*time_correction_factor - t_offset
                logger.warning(f'Applied time axis scale correction factor {time_correction_factor}: '
                               f'{time_correction_info}')
            # raise ValueError(message)


    # Use origin information from reading movie meta data to read frame data from same data source (speed & consistency)
    # movie_path = [movie_origin.get('path', None)]
    # movie_fn = [movie_origin.get('fn', None)]
    # movie_plugin = {movie_origin['plugin']: movie_plugins[movie_origin['plugin']]}
    # (frame_nos, frame_times, frame_data), movie_origin = plugins.plugins_movie.read_movie_data(pulse, camera, machine,
    #                                                             movie_plugin,movie_paths=movie_path, movie_fns=movie_fn,
    #                                                             verbose=True)

    # Detect saturated pixels, uniform frames etc
    bad_frames_info = data_quality.identify_bad_frames(frame_data, bit_depth=movie_meta['bit_depth'],
                                                       n_discontinuities_expected=1, raise_on_saturated=False)
    frames_nos_discontinuous = bad_frames_info['discontinuous']['frames']['n']
    frame_data_clipped, removed_frames = data_quality.remove_bad_opening_and_closing_frames(frame_data,
                                                                                    frames_nos_discontinuous)
    # TODO: Detect frames with non-uniform time differences (see heat flux func)

    # Now frame_data is in its near final form (bad frames removed) merge it into image_data
    image_data = xr.merge([image_data, frame_data_clipped])  # , combine_attrs='no_conflicts')

    # Use dark parts of image to detect dark level drift
    # TODO: Store dark level correction and mask in dataset
    dark_level, dark_level_correction_factors, mask_dark = camera_checks.get_dark_level_drift(image_data)

    apply_dark_level_correction = True
    # apply_dark_level_correction = False
    if apply_dark_level_correction:
        frame_data = camera_checks.correct_dark_level_drift(image_data['frame_data'], dark_level_correction_factors)
        image_data['frame_data'] = frame_data

    if (debug.get('dark_level', False)):
        camera_checks.plot_dark_level_variation(mask_dark=mask_dark, frame_data=image_data['frame_data'],
                                                dark_level=dark_level)

    t = image_data['t']

    if (debug.get('movie_intensity_stats-corrected', False) or
            ((not scheduler) and (bad_frames_info['saturated']['n_bad_frames']))):
        # Force plot if there are any saturated frames
        fig, ax, ax_n = temporal_figures.plot_temporal_stats(image_data['frame_data'], meta_data=meta_data,
                                                             num='movie_intensity_stats-corrected')
        if removed_frames:
            # Label ends of movie that are discarded due discontinuous intensities etc
            ax_n.axvline(x=removed_frames['start'], ls=':', color='k', label='clipped bad frames from start')
            ax_n.axvline(x=len(image_data['frame_data']) - 1 - removed_frames['end'], ls=':', color='k',
                         label='clipped bad frames from end')
        plot_tools.show_if(show=True)

    # TODO: Add coordinates for pre-transformation frame data
    # image_data_ref = xr.Dataset(coords=image_data.coords)
    # image_data_ref.coords['n_ref'] = ('n_ref', [n_middle])
    # image_data_ref['frame_ref_pre_transforms'] = (('n_ref', 'y_pix_raw', 'x_pix_raw'),
    #                                               frame_ref_pre_transforms[np.newaxis, ...])

    # TODO: Lookup and apply t_offset correction to frame times? See IDL get_toffset.pro reading toffset.dat

    # Fix camera shake
    if analaysis_steps['camera_shake_correction']:
        # TODO: Find best reference frame, as want bright clear frame with NUC shutter
        # TODO: Consider checking camera rotation which is issue on ASDEX-U
        percentile_n_shake_ref = 10
        # percentile_n_shake_ref = 50
        # percentile_n_shake_ref = 85
        n_shake_ref = np.percentile(frame_data.n, percentile_n_shake_ref, interpolation='nearest')  # Frame towards video end
        # n_shake_ref = frame_data.mean(dim=('x_pix', 'y_pix')).argmax(dim='n')  # brightest frame (may have shake?)
        frame_shake_ref = image_data['frame_data'][n_shake_ref]
        # frame_shake_ref = calcam_calib_image_windowed
        erroneous_displacement = 50
        # TODO: Apply for multiple camera shake ref frames to account for variation across movie (take min displacements)
        pixel_displacements, shake_stats = camera_shake.calc_camera_shake_displacements(image_data['frame_data'],
                                        frame_shake_ref,erroneous_displacement=erroneous_displacement, verbose=True)
        pixel_displacements = xr.DataArray(pixel_displacements, coords={'n': image_data['n'], 'pixel_coord': ['x', 'y']},
                                          dims=['n', 'pixel_coord'])
        image_data['pixel_displacements'] = pixel_displacements
        frame_data = camera_shake.remove_camera_shake(image_data['frame_data'], pixel_displacements=pixel_displacements,
                                                      verbose=True)
        image_data['frame_data'] = frame_data

        if debug.get('camera_shake', False):
            debug_plots.debug_camera_shake(pixel_displacements=pixel_displacements, n_shake_ref=n_shake_ref)

    # TODO: Pass bad_frames_info to debug plots

    # TODO: Lookup known anomalies to mask - eg stray LP cable, jammed shutter - replace image data with nans?

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
    # TODO: consider using custon nuc time/frame number range for each movie? IDL sched had NUC time option
    # TODO: Look up first shot of day and load starting nuc frames from there to avoid warming from preceding shots
    nuc_frame = nuc.get_nuc_frame(origin={'n': nuc_frame_range}, frame_data=image_data['frame_data'],
                                  reduce_func='mean')

    # nuc_frame = get_nuc_frame(origin={'n': [None, None]}, frame_data=frame_data, reduce_func='min')
    frame_data_nuc = nuc.apply_nuc_correction(image_data['frame_data'], nuc_frame, raise_on_negatives=False)
    frame_data_nuc = data_structures.attach_standard_meta_attrs(frame_data_nuc, varname='frame_data_nuc',
                                                                key='frame_data')
    image_data['nuc_frame'] = (('y_pix', 'x_pix'), nuc_frame.data)
    image_data['frame_data_nuc'] = frame_data_nuc

    if (debug.get('movie_intensity_stats-nuc', False) or
            ((not scheduler) and (bad_frames_info['saturated']['n_bad_frames']))):
        # Force plot if there are any saturated frames
        fig, ax, ax_n = temporal_figures.plot_temporal_stats(image_data['frame_data_nuc'], meta_data=meta_data,
                                                             num='movie_intensity_stats-nuc')
        if removed_frames:
            # Label ends of movie that are discarded due discontinuous intensities etc
            ax_n.axvline(x=removed_frames['start'], ls=':', color='k', label='clipped bad frames from start')
            ax_n.axvline(x=len(image_data['frame_data_nuc']) - 1 - removed_frames['end'], ls=':', color='k',
                         label='clipped bad frames from end')
        plot_tools.show_if(show=True)

    if output.get('raw_frame_image', False):
        # nuc_out = True
        nuc_out = False
        n_ref = 240
        path_fn = paths_output['raw_images'] / f'{machine}-{camera}-{pulse}-n{n_ref}{"_nuc"*nuc_out}.png'
        key = 'frame_data_nuc' if nuc_out else 'frame_data'
        image_figures.figure_frame_data(image_data, n=n_ref, key=key, label_outliers=False,
                                            axes_off=True, show=True, save_fn_image=path_fn)

    if debug.get('debug_detector_window', False):  # Need to plot before detector window applied to calibration
        debug_plots.debug_detector_window(detector_window=detector_window, frame_data=image_data,
                                          calcam_calib=calcam_calib, image_full_frame=calcam_calib_image_full_frame,
                                          image_coords=image_coords)
    if debug.get('movie_data_animation', False):
        image_figures.animate_frame_data(image_data, key='frame_data', nth_frame=1)
                        #                  n_start=40, n_end=350,
                        # save_path_fn=paths_output['gifs'] / f'{machine}-{pulse}-{camera}-frame_data.gif')
    if debug.get('movie_data_nuc_animation', False):
        image_figures.animate_frame_data(image_data, key='frame_data_nuc', nth_frame=1)

    if debug.get('movie_data_nuc', False):
        debug_plots.debug_movie_data(image_data, key='frame_data_nuc')

    if debug.get('specific_frames', False):
        n_check = 218
        debug_plots.debug_movie_data(image_data, frame_nos=np.arange(n_check, n_check+4), key='frame_data')
        # frame_data_nuc

    if debug.get('calcam_calib_image', False):
        debug_plots.debug_calcam_calib_image(calcam_calib, frame_data=frame_data,
                                             frame_ref=frame_ref_pre_transforms, n_frame_ref=n_middle,
                                             wire_frame=image_data['wire_frame'])

    if debug.get('spatial_res', False):
        debug_plots.debug_spatial_res(image_data)

    if figures.get('spatial_res', False):
        spatial_res_type = 'max'
        # spatial_res_type = 'y'
        clip_range = [2.5, 20]
        # clip_range = [1.5, 3.0]  # air
        fn_spatial_res = path_figures / f'spatial_res_{pulse}_{camera}_{spatial_res_type}.png'
        image_figures.figure_spatial_res(image_data, res_type=spatial_res_type, clip_range=clip_range, log_cmap=False,
                                         save_fn=fn_spatial_res, show=True)

    # TODO: calculate rho_psi coordinate across image
    # TODO: Include strike point number eg snow flake SP4

    # TODO: call plugin function to get s coordinate along tiles?
    # s_im = get_s_coord_global(x_im, y_im, z_im, machine_plugins)
    # data['s_im'] = (('y_pix', 'x_pix'), s_im)

    if False and debug.get('spatial_coords', False):
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
        # points_pix plotted as green crosses
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
    # surface_coords = io_basic.read_csv(files['structure_coords'], sep=', ', index_col='structure')
    _, surface_coords = geometry.read_structure_coords(path_fn=files['structure_coords'], machine=machine)
    structure_coords_df, structure_coords_info = geometry.structure_coords_dict_to_df(surface_coords)
    r_im, phi_im, z_im = image_data['R_im'], image_data['phi_im'], image_data['z_im']
    visible_surfaces = geometry.identify_visible_structures(r_im, phi_im, z_im, structure_coords_df, phi_in_deg=False)
    surface_ids, material_ids, visible_surfaces, visible_materials = visible_surfaces

    image_data['surface_id'] = (('y_pix', 'x_pix'), surface_ids)
    image_data['material_id'] = (('y_pix', 'x_pix'), material_ids)
    image_data.attrs['visible_surfaces'] = visible_surfaces
    image_data.attrs['visible_materials'] = visible_materials

    if debug.get('surfaces', False):
        debug_plots.debug_surfaces(image_data)

    # Read thermal properties of materials for structures in view
    material_names = list(set(visible_materials.values()))
    material_properties = interfaces.json_load(files['material_props'], key_paths_keep=material_names,
                                               key_paths_drop=('README',), lists_to_arrays=True)
    image_data.attrs['material_properties'] = material_properties
    # TODO: Segment path according to changes in tile properties

    calib_coefs = lookup_info['temperature_coefs'].to_dict()
    # calib_coefs = lookup_pulse_row_in_csv(files['calib_coefs'], pulse=pulse, header=4)
    # TODO: Switch to plugin for temperature calculation if methods on different machines can't be standardised?
    if False and (machine == 'mast'):  # pulse < 40000:
        frame_data_nuc[0, :, :] = 0  # set first frame to zero for consistency with IDL code which init frame(0) to 0
        # TODO: Remove legacy temperature method with old MAST photon count lookup table
        files['black_body_curve'] = Path('/home/tfarley/repos/air/fire/input_files/mast/legacy/legacy_air/planckBB.dat')
        bb_curve = io_basic.read_csv(files['black_body_curve'], sep=r'\s+',
                                     names=['temperature_celcius', 'photon_flux'], index_col='temperature_celcius')
        image_data['temperature_im'] = temperature.dl_to_temerature_legacy(frame_data_nuc, calib_coefs, bb_curve,
                                                           exposure=movie_meta['exposure'],
                                                           solid_angle_pixel=meta_data['solid_angle_pixel'],
                                                           trans_correction=1.61,
                                                           temperature_bg_nuc=temp_bg, meta_data=meta_data['variables'])
    else:
        # TODO: Update camera settings input files with correct window transmittance for known window numbers
        # TODO: Pass in solid_angle_pixel (requires modifying calibration coeffs as produced using 2 pi)
        image_data['temperature_im'] = temperature.dl_to_temerature(frame_data_nuc, calib_coefs,
                                                    wavelength_range=camera_settings['wavelength_range'],
                                                    integration_time=movie_meta['exposure'],
                                                    transmittance=lookup_info['camera_settings']['window_transmission'],
                                                    solid_angle_pixel=2*np.pi,  # meta_data['solid_angle_pixel'],
                                                    temperature_bg_nuc=temp_bg,
                                                    lens_focal_length=meta_data['lens'],
                                                    meta_data=meta_data['variables'])
    image_data = data_structures.attach_standard_meta_attrs(image_data, varname='temperature_im', key='temperature',
                                                            replace=True)

    # TODO: Identify hotspots: MOI 3.2: Machine protection peak tile surface T from IR is 1300 C
    # (bulk tile 250C from T/C)

    if debug.get('temperature_im', False):
        debug_plots.debug_temperature_image(image_data)
        
    if debug.get('movie_temperature_animation', False):
        # save_path_fn = paths_output['gifs'] / f'{machine}-{camera}-{pulse}_temperature_movie.gif'
        # save_path_fn = path_figures / f'rit_{pulse}_temperature_movie.gif'
        save_path_fn = None
        frame_range = [40, 410]
        cbar_range = [0, 99.9]  # percentage of range
        # cbar_range = None
        frame_range = np.clip(frame_range, *meta_data['frame_range'])

        image_figures.animate_frame_data(image_data, key='temperature_im', nth_frame=1, duration=15,
                                         n_start=frame_range[0], n_end=frame_range[1], save_path_fn=save_path_fn,
                                         cbar_range=cbar_range,
                                         frame_label=f'{camera.upper()} {pulse} $t=${{t:0.1f}} ms',
                                         cbar_label='$T$ [$^\circ$C]',
                                         label_values={'t': image_data['t'].values * 1e3}, show=True)

    # TODO: Calculate toroidally averaged radial profiles taking into account viewing geometry
    # - may be more complicated than effectively rotating image slightly as in MAST (see data in Thornton2015)

    # TODO: Temporal smoothing of temperature

    # Get spatial coordinates of analysis paths
    # TODO: get material, structure, bad pixel ids along analysis path
    analysis_path_dfn_points_all = interfaces.json_load(files['analysis_path_dfns'], key_paths_drop=('README',))

    # Get names of analysis paths to use in this analysis. Multiple analysis path names in csv can be separated with ;'s
    analysis_path_names = lookup_info['analysis_paths']['analysis_path_name'].split(';')
    analysis_path_names = {f'path{i}': name.strip() for i, name in enumerate(analysis_path_names)}
    analysis_path_keys = list(analysis_path_names.keys())
    analysis_path_labels = list(analysis_path_names.values())  # initialised value to be replaced with values from file

    meta_data['analysis_path_names'] = list(analysis_path_names.values())
    meta_data['analysis_path_keys'] = analysis_path_keys
    meta_data['analysis_path_labels'] = analysis_path_labels

    # TODO: If required project spatial analysis paths into image analysis path definitions here (ie make spatial dfn
    # optional)

    # path_data_all = xr.Dataset()
    for i_path, (analysis_path_key, analysis_path_name) in enumerate(analysis_path_names.items()):
        path = analysis_path_key

        path_coord = f'i_{analysis_path_key}'
        analysis_path_dfn = analysis_path_dfn_points_all[analysis_path_name]
        # TODO: Move standardisation to separate loop before analysis loop?
        analysis_path_dfn, analysis_path_dfn_dict = calcam_calibs.standardise_analysis_path_definition(
                        analysis_path_dfn, calcam_calib, analysis_path_key, image_coords=image_coords)
        # TODO: store record of analysis_path_dfn to preserve after next loop

        path_label = analysis_path_dfn_dict.get('label', analysis_path_name)
        analysis_path_labels[i_path] = path_label
        meta_data['path_label'] = path_label  # tmp for plotting TODO: Remove?

        logger.info(f'Performing analysis along analysis path "{analysis_path_name}" '
                    f'("{path_label}" / "{analysis_path_key}")')

        analysis_path_dfn_points = analysis_path_dfn_dict['coords']
        analysis_path_description = analysis_path_dfn_dict['description']

        # Get interpolated pixel and spatial coords along path
        frame_masks = {'surface_id': surface_ids, 'material_id': material_ids}

        # TODO: Test with windowing
        path_data = calcam_calibs.join_analysis_path_control_points(analysis_path_dfn, analysis_path_key,
                                                                    frame_masks, image_shape)
        # NOTE: meta_data['image_shape'] and ipx header hight/width etc prior to calcam image transformations
        # image_data = xr.merge([image_data, path_data])  # , combine_attrs='no_conflicts')

        path_data_extracted = image_processing.extract_path_data_from_images(image_data, path_data,
                                                                             path_name=analysis_path_key)
        path_data = xr.merge([path_data, path_data_extracted])  # , combine_attrs='no_conflicts')

        # Sort analysis path in order of increasing R to avoid reversing sections of profiles!
        path_data = path_data.sortby('R_in_frame_path0', ascending=True)  # TODO: Detect non-mono and tidy into func?
        # path_data = path_data.sortby('s_global_in_frame_path0', ascending=True)  # TODO: Detect non-mono and tidy into
        # func?

        # TODO: itteratively remove negative and zero increments in R along path. Enables use of dataarray.sel(
        # method='nearest')

        path_data.attrs['meta'] = meta_data

        # TODO: Filter nan spatial coordinates from analysis path?
        missing_material_key = -1  # Set to None to reproduce legacy MAST analysis
        # missing_material_key = None  # Set to None to reproduce legacy MAST analysis
        if missing_material_key is None:
            logger.warning('missing_material_key = None to reproduce legacy MAST analysis')
        path_data = image_processing.filter_unknown_materials_from_analysis_path(path_data, analysis_path_key,
                                                                    missing_material_key=missing_material_key)

        visible_surfaces_path = geometry.identify_visible_structures(path_data[f'R_{path}'],
                                    path_data[f'phi_deg_{path}'], path_data[f'z_{path}'], structure_coords_df,
                                                                       phi_in_deg=True)
        surface_ids_path, material_ids_path, visible_surfaces_path, visible_materials_path = visible_surfaces_path
        logger.info(f'Surfaces visible along path "{analysis_path_name}": {visible_surfaces_path}')

        # Add local spatial resolution along path
        s_local = s_coordinate.calc_local_s_along_path_data_array(path_data, analysis_path_key)
        path_data = xr.merge([path_data, s_local])  # , combine_attrs='no_conflicts')
        coord = f's_local_{path}'
        attrs = path_data[coord].attrs  # attrs get lost in conversion to coordinate
        path_data = path_data.assign_coords(**{coord: (f'i_{path}', path_data[coord].values)})  # Make coord for plots
        path_data[coord].attrs.update(attrs)
        path_data.attrs['meta'] = meta_data

        # x_path, y_path, z_path = (path_data[f'{coord}_path'] for coord in ['x', 'y', 'z'])
        # s_path = get_s_coord_path(x_path, y_path, z_path, machine_plugins)
        # path_data['s_path'] = s_path

        if debug.get('poloidal_cross_sec', False):
            spatial_figures.figure_poloidal_cross_section(image_data=image_data, path_data=path_data, pulse=pulse,
                                                          no_cal=True, show=True)

        if debug.get('path_cross_sections', False):
            debug_plots.debug_analysis_path_cross_sections(path_data, image_data=image_data,
                        path_names=analysis_path_key, image_data_in_cross_sections=True,
                            machine_plugins=machine_plugins, pupil_coords=calcam_calib.get_pupilpos(), show=True)

        if debug.get('spatial_coords', False):
            debug_plots.debug_spatial_coords(image_data, path_data=path_data, path_name=analysis_path_key,
                                             coord_keys=('x_im', 'y_im',
                                                        'R_im', 'phi_deg_im', 'z_im',
                                                        'ray_lengths_im', 's_global_im', 'wire_frame'))

        if debug.get('temperature_vs_R_t', False):
            debug_plots.debug_plot_profile_2d(path_data, param='temperature', path_names=analysis_path_key,
                                              num='temperature_vs_R_t', robust=False, meta=meta_data,
                                              machine_plugins=machine_plugins, verbose=True)

        force_material_sub_index = (None if (missing_material_key == -1) else 1)
        material_name, theo_kwargs = heat_flux_module.theo_kwargs_for_path(material_ids, visible_materials, material_properties,
                                                force_material_sub_index=force_material_sub_index)
        meta_data['theo_kwargs'] = theo_kwargs
        meta_data['theo_visible_material'] = material_name

        heat_flux, extra_results = heat_flux_module.calc_heatflux(image_data['t'], image_data['temperature_im'],
                                                                  path_data, analysis_path_key, theo_kwargs,
                                                                  force_material_sub_index=force_material_sub_index,
                                                                  meta=meta_data)

        # TODO: Consolidate variable_meta_dat from fire_config.json and data_structures.py
        heat_flux_key = f'heat_flux_{analysis_path_key}'
        path_data[heat_flux_key] = (('t', path_coord), heat_flux.T)
        path_data = data_structures.attach_standard_meta_attrs(path_data, heat_flux_key, key='q')

        # path_data[heat_flux_key].attrs.update(meta_data['variables'].get('heat_flux', {}))
        # if 'description' in path_data[heat_flux_key].attrs:
        #     path_data[heat_flux_key].attrs['label'] = path_data[heat_flux_key].attrs['description']
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

        path_data_all = xr.merge([path_data_all, path_data])  # , combine_attrs='no_conflicts')

        if debug.get('heat_flux_vs_R_t', False):
            # robust = True
            robust = False
            extend = None
            debug_plots.debug_plot_profile_2d(path_data, param='heat_flux', path_names=analysis_path_key,
                                              robust=robust, extend=extend, meta=meta_data, mark_peak=True,
                                              machine_plugins=machine_plugins)
        if figures.get('heat_flux_vs_R_t-robust', False):
            robust = True
            extend = 'both'
            fn = f'heat_flux_vs_R_t-robust-{machine}_{camera}_{pulse}.png'
            save_path_fn = path_figures / 'heat_flux_vs_R_t-robust' / fn
            debug_plots.debug_plot_profile_2d(path_data, param='heat_flux', path_names=analysis_path_key,
                                              extend=extend, robust=robust, meta=meta_data,
                                              num='heat_flux_vs_R_t-robust',  machine_plugins=machine_plugins,
                                              show=False, save_path_fn=save_path_fn)

        if debug.get('analysis_path', False):
            # TODO: Finish  debug_analysis_path_2d
            # debug_plots.debug_analysis_path_2d(image_data, path_data, path_names=analysis_path_key,
            #                        image_data_in_cross_sections=True, machine_plugins=machine_plugins)
            debug_plots.debug_analysis_path_1d(image_data, path_data, path_names=analysis_path_key,
                           image_data_in_cross_sections=True, machine_plugins=machine_plugins,
                           pupil_coords=calcam_calib.get_pupilpos(),
                           keys_profiles=(
                           ('frame_data_min(i)_{path}', 'frame_data_mean(i)_{path}', 'frame_data_max(i)_{path}'),
                           ('temperature_min(i)_{path}', 'temperature_mean(i)_{path}', 'temperature_max(i)_{path}'),
                           ('heat_flux_min(i)_{path}', 'heat_flux_mean(i)_{path}', 'heat_flux_max(i)_{path}'),
                           ('s_global_{path}', 'R_{path}'),
                           ('ray_lengths_{path}',),
                           ('spatial_res_x_{path}', 'spatial_res_y_{path}', 'spatial_res_linear_{path}')
                           ))

        # if debug.get('heat_flux_path_1d', False):
        #     debug_plots.debug_analysis_path_1d(image_data, path_data, path_names=analysis_path_key,
        #                    image_data_in_cross_sections=False, machine_plugins=machine_plugins,
        #                    pupil_coords=calcam_calib.get_pupilpos())

        if debug.get('timings', False):
            # TODO: Use correct clock signal for camera
            debug_plots.debug_plot_timings(path_data_all, pulse=pulse, meta_data=meta_data)

        if debug.get('strike_point_loc', False):
            heat_flux = path_data[f'heat_flux_amplitude_peak_global_{path}'].values
            heat_flux_thresh = np.nanmin(heat_flux) + 0.03 * (np.nanmax(heat_flux) - np.nanargmin(heat_flux))
            debug_plots.debug_plot_temporal_profile_1d(path_data_all, params=('heat_flux_R_peak', 'heat_flux_amplitude_peak_global'),
                                                       path_name=analysis_path_key, x_var='t',
                                                       heat_flux_thresh=heat_flux_thresh, meta_data=meta_data)
        if output.get('strike_point_loc', False):
            fn = f'strike_point_loc-{machine}-{camera}-{pulse}-{analysis_path_name}.csv'
            path_fn = Path(paths_output['csv_data']) / fn
            io_basic.to_csv(path_fn, path_data, cols=f'heat_flux_R_peak_{path}', index='t', x_range=[0, 0.6],
                            drop_other_coords=True, verbose=True)

    meta_data['analysis_path_labels'] = analysis_path_labels

    # TODO: to be compatible with xpad the coords have to be increasing hence if sign is negative we have to reverse the
    # zcoord and reverse the tprofile and qprofile
    # path_fn_out = files['processed_ir_netcdf']

    # TODO: Identify variables/coords with empty attrs and attach standard meta data
    # TODO: Separate output data selection from ouput code - pass in single dataset, all of which should be written
    # TODO: Define standardised interface for outputs
    # TODO: Pass absolute path to output file?
    output_header_info = config['outputs']['file_header']
    raise_on_fail = False
    # raise_on_fail = False
    device_details = utils.add_aliases_to_dict(device_details,
                                        aliases=dict(camera_serial_number='serial', detector_resolution='resolution'),
                                               remove_original=False)

    outputs = plugins.plugins_output_format.write_to_output_format(output_plugins, path_data=path_data_all,
                                                                   image_data=image_data,
                           path_names=analysis_path_keys, variable_names_path=output_variables['analysis_path'],
                           variable_names_time=output_variables['time'], variable_names_image=output_variables['image'],
                           device_info=device_details, header_info=output_header_info, meta_data=meta_data,
                                                                   raise_on_fail=raise_on_fail, client=client)
    # write_processed_ir_to_netcdf(data, path_fn_out)

    archive_netcdf_output = True
    # archive_netcdf_output = False
    if archive_netcdf_output:
        path_fn_netcdf = outputs.get('uda_putdata', {}).get('path_fn')
        interfaces.archive_netcdf_output(path_fn_netcdf, meta_data=meta_data)

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
    debug = {'debug_detector_window': False, 'camera_shake': True,
             'movie_data_animation': False, 'movie_data_nuc_animation': False,
             'spatial_coords': True, 'spatial_res': False, 'movie_data_nuc': False,
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

    # pulse = 28623  # 700kA

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
             'movie_data_animation': False, 'movie_data_nuc_animation': False,
             'spatial_coords': True, 'spatial_res': False, 'movie_data_nuc': False,
             'surfaces': False, 'analysis_path': True, 'temperature_im': False}
    # debug = {k: True for k in debug}
    debug = {k: False for k in debug}
    figures = {'spatial_res': False}
    logger.info(f'Running MAST scheduler workflow...')
    status = scheduler_workflow(pulse=pulse, camera=camera, pass_no=pass_no, machine=machine, scheduler=scheduler,
                       equilibrium=magnetics, update_checkpoints=update_checkpoints, debug=debug, figures=figures)
    return status

def run_mast_rit():  # pragma: no cover
    # pulse = 30378
    # pulse = 29936
    # pulse = 27880  # Full frame upper divertor
    # pulse = 29000  # 80x256 upper divertor
    # pulse = 28623   # 700kA - no data?
    pulse = 26798   # MAST 400kA
    camera = 'rit'

    pass_no = 0
    machine = 'MAST'
    scheduler = False
    magnetics = False
    update_checkpoints = False
    # update_checkpoints = True
    debug = {'debug_detector_window': True, 'camera_shake': True,
             'movie_data_animation': False, 'movie_data_nuc_animation': False,
             'spatial_coords': True, 'spatial_res': False, 'movie_data_nuc': False,
             'surfaces': False, 'analysis_path': True, 'temperature_im': False,}
    # debug = {k: True for k in debug}
    debug = {k: False for k in debug}
    figures = {'spatial_res': False}
    logger.info(f'Running MAST scheduler workflow...')
    status = scheduler_workflow(pulse=pulse, camera=camera, pass_no=pass_no, machine=machine, scheduler=scheduler,
                       equilibrium=magnetics, update_checkpoints=update_checkpoints, debug=debug, figures=figures)
    return status

def run_mastu_rir():  # pragma: no cover
    # pulse = 50000  # CAD view calibrated from test installation images - no plasma
    # pulse = 50001  # Test movie consisting of black body cavity calibration images

    # pulse = 44726  #
    # pulse = 44673  # DN-700-SXD-OH
    pulse = 43952  # early focus

    camera = 'rir'
    pass_no = 0
    machine = 'MAST_U'

    scheduler = False
    magnetics = False
    update_checkpoints = False
    # update_checkpoints = True

    # TODO: Remove redundant movie_data step
    debug = {'calcam_calib_image': False, 'debug_detector_window': False,
             'movie_intensity_stats': True,
             'movie_data_animation': False, 'movie_data_nuc_animation': False,
             'movie_temperature_animation': False,
             'spatial_coords': True,
             'spatial_res': False,
             'movie_data_nuc': True, 'specific_frames': False, 'camera_shake': False, 'temperature_im': False,
             'surfaces': False, 'analysis_path': False,
             'path_cross_sections': False,
             'temperature_vs_R_t': True,
             'heat_flux_vs_R_t': True,
             'timings': True,
             'strike_point_loc': True,
             # 'heat_flux_path_1d': True,
             }

    output = {'strike_point_loc': True, 'raw_frame_image': False}

    # debug = {k: True for k in debug}
    # debug = {k: False for k in debug}
    figures = {'spatial_res': False, 'heat_flux_vs_R_t-robust': True}
    logger.info(f'Running {machine} {camera} scheduler workflow...')
    status = scheduler_workflow(pulse=pulse, camera=camera, pass_no=pass_no, machine=machine, scheduler=scheduler,
                       equilibrium=magnetics, update_checkpoints=update_checkpoints, debug=debug, figures=figures,
                                output=output)
    return status

def run_mastu_rit():  # pragma: no cover
    # pulse = 50000  # CAD view calibrated from test installation images - no plasma
    # pulse = 50001  # Test movie consisting of black body cavity calibration images
    # pulse = 50002  # IRCAM Works raw file for debugging
    # pulse = 43141  # Early diverted plasma on T2-T4
    # pulse = 43183  # Early diverted plasma on T2-T5
    # pulse = 43163  # TODO: check frame rate and expsure meta data on LWIR PC
    # pulse = 43412  # Peter Ryan's strike point sweep based on 43391 for LP checks
    # pulse = 43413  # Peter Ryan's strike point sweep based on 43391 for LP checks
    # pulse = 43415  # Peter Ryan's strike point sweep for LP checks
    # pulse = 43524  # Double NBI in
    # pulse = 43530  # Peter Ryan's strike point sweep for LP checks
    # pulse = 43534  # T5 sweep
    # pulse = 43535  # T5 sweep with NBI
    # pulse = 43547  # T5 sweep with NBI
    # pulse = 43561  # NBI
    # pulse = 43575  # NBI
    # pulse = 43583  # NBI
    # pulse = 43584  # NBI
    # pulse = 43591
    # pulse = 43587
    # pulse = 43610
    # pulse = 43643
    # pulse = 43644
    # pulse = 43648
    # pulse = 43685

    # pulse = 43610  # MU KPI
    # pulse = 43611
    # pulse = 43613
    # pulse = 43614
    # pulse = 43591
    # pulse = 43596
    # pulse = 43415
    # pulse = 43644
    # pulse = 43587

    # Peter's list of shots with a strike point sweep to T5:
    # pulse = 43756  # LP, but NO IR data
    # pulse = 43755  # NO LP or IR data
    # pulse = 43529  # LP, but NO IR data
    # pulse = 43415  # LP and IR data --
    # pulse = 43412  # LP and IR data --
    # pulse = 43391  # no LP or IR data

    # pulse = 43753  # Lidia strike point splitting request - no data

    # pulse = 43805  # Strike point sweep to T5 - good data for IR and LP

    # pulse = 43823
    # pulse = 43835  # Lidia strike point splitting request - good data
    # pulse = 44004  # LM
    # pulse = 43835
    # pulse = 43852
    # pulse = 43836

    # pulse = 43839
    # pulse = 43996

    # pulse = 43998  # Super-X

    # pulse = 44463  # first irircam automatic aquisition

    # pulse = 44386

    # pulse = 44628  # DATAC software acquisition ftp'd straight to data store - access via UDA
    # pulse = 44697  # DATAC software acquisition ftp'd straight to data store - access via UDA

    # pulse = 44677   # RT18 - Swept (attached?) super-x
    pulse = 44683   # RT18 - Attached T5 super-x

    # pulse = 44695   # 400 kA
    # pulse = 44613   # 400 kA

    # pulse = 44550   # error field exp
    # pulse = 44776   # error field exp
    # pulse = 44777   # error field exp
    # pulse = 44778   # error field exp
    # pulse = 44779   # error field exp

    # pulse = 44815   # RT13

    # pulse = 44852  #  TF test shot with gas

    # pulse = 44677  # Standard pulse JH suggests comparing with all diagnostics - RT18 slack, time to eurofusion

    # 44849 onwards should have uda efit

    camera = 'rit'
    pass_no = 0
    machine = 'MAST_U'

    scheduler = False
    magnetics = False
    update_checkpoints = False
    # update_checkpoints = True

    # TODO: Remove redundant movie_data step
    debug = {'calcam_calib_image': False, 'debug_detector_window': False,
             'movie_intensity_stats-raw': True,
             'movie_intensity_stats-corrected': True,
             'movie_intensity_stats-nuc': True,
             'dark_level': True,
             'movie_data_animation': False, 'movie_data_nuc_animation': False,
             'movie_temperature_animation': False,
             'spatial_coords': False,
             'spatial_res': False,
             'movie_data_nuc': False, 'specific_frames': False, 'camera_shake': False, 'temperature_im': False,
             'surfaces': False, 'analysis_path': False,
             'path_cross_sections': False,
             'temperature_vs_R_t': False,
             'heat_flux_vs_R_t': True,
             'timings': True,
             'strike_point_loc': False,
             # 'heat_flux_path_1d': True,
             }

    output = {'strike_point_loc': True, 'raw_frame_image': False}

    # debug = {k: True for k in debug}
    # debug = {k: False for k in debug}
    figures = {'spatial_res': False, 'heat_flux_vs_R_t-robust': True}
    logger.info(f'Running MAST-U ait scheduler workflow...')
    status = scheduler_workflow(pulse=pulse, camera=camera, pass_no=pass_no, machine=machine, scheduler=scheduler,
                       equilibrium=magnetics, update_checkpoints=update_checkpoints, debug=debug, figures=figures,
                                output=output)
    return status

if __name__ == '__main__':
    # delete_file('~/.fire_config.json', verbose=True, raise_on_fail=True)
    copy_default_user_settings(replace_existing=True)

    # AIR, AIT, AIS, AIU, AIV
    # status = run_mast_rir()
    # status = run_mast_rit()
    # status = run_mastu_rir()
    status = run_mastu_rit()
    # status = run_jet()

    outputs = status['outputs']

    clean_netcdf = True
    copy_to_uda_scrach = True

    path_fn_netcdf = outputs.get('uda_putdata', {}).get('path_fn')

    if copy_to_uda_scrach and (path_fn_netcdf is not None):
        path_fn_scratch = Path('/common/uda-scratch/IR') / path_fn_netcdf.name
        path_fn_scratch.parent.mkdir(exist_ok=True)
        path_fn_scratch.write_bytes(path_fn_netcdf.read_bytes())
        logger.info(f'Copied uda output file to uda scratch: {str(path_fn_scratch)}')

    if clean_netcdf:
        if ('uda_putdata' in outputs) and (outputs['uda_putdata']['success']):

            if path_fn_netcdf.is_file():
                io_basic.copy_file(path_fn_netcdf, path_fn_scratch)

                path_fn_netcdf.unlink()
                logger.info(f'Deleted uda output file to avoid clutter since run from main in scheduler_workflow.py: '
                            f'{path_fn_netcdf}')