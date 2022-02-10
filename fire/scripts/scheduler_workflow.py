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
from fire import plugins, camera_tools
from fire.interfaces import interfaces, calcam_calibs, io_basic, io_utils, uda_utils, user_config
from fire.camera_tools import field_of_view, camera_shake, nuc, image_processing, camera_checks
from fire.geometry import geometry, s_coordinate
from fire.physics import temperature, heat_flux, physics_parameters, efit_utils
from fire.misc import data_quality, data_structures, utils
from fire.plotting import plot_tools, debug_plots, image_figures, temporal_figures, spatial_figures
heat_flux_module = heat_flux  # Alias to avoid name clashes

# TODO: remove after debugging core dumps etc
# import faulthandler
# faulthandler.enable()

# TODO: Remove after debugging pandas warnings
import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

# logger = logging.getLogger(__name__)  # TODO: Check propagation of fire_logger file handler
logger = logging.getLogger('fire.scheduler_workflow')  # TODO: Check propagation of fire_logger file handler
# logger.setLevel(logging.DEBUG)
# print(logger_info(logger))

cwd = Path(__file__).parent
# path_figures = (cwd / '../figures/').resolve()
# path_figures = (fire_paths['user'] / 'figures/').resolve()
# logger.info(f'Scheduler workflow: Figures will be output to: {path_figures}')

def scheduler_workflow(pulse:Union[int, str], camera:str='rir', pass_no:int=0, machine:str='MAST-U',
                       scheduler:bool=False,
                       equilibrium:bool=False, update_checkpoints:bool=False, analysis_steps: Optional[list]=None,
                       n_cores=1,
                       path_out=None, path_calib=None, path_user=None,
                       movie_plugins_filter=None,
                       image_coords='Display',
                       debug: Optional[dict]=None, figures: Optional[dict]=None, output_files: Optional[dict]=None):
    """ Primary analysis workflow for MAST-U/JET IR scheduler analysis.
    Called from the command line with run_fire.py.

    Args:
        pulse               : Shot/pulse number or string name for synthetic movie data
        camera              : Name of camera to analyse (unique name of camera or diagnostic tag)
        pass_no             : Scheduler pass number
        machine             : Tokamak/device that the data originates from
        shceduler           : (Optional) Has code been called by the scheduler?
        equilibrium         : (Optional) Produce additional output with scheduler efit as additional dependency
        update_checkpoints  : (Optional) Update (eg calcam geometry) checkpoint files used to speed up repeated analysis
        analysis_steps      : (Optional) Which analysis steps to perform
        n_cores             : (Optional) Number of cores to use for multiprocessing
        path_out            : (Optional) User supplied path for (default eg UDA) output file
        path_calib          : (Optional) User supplied path for calibration input files
        path_user           : (Optional) User supplied path for user config file and output (eg figures)
        movie_plugins_filter: (Optional) List of movie plugin names to restrict use to
        image_coords        : (Optional) Whether to perform analysis in 'Display' or 'Original' coords (see Calcam defn)
        debug               : (Optional) Dict specifying which debug figures to plot
        figures             : (Optional) Dict specifying which output figures to plot/save
        output_files        : (Optional) Dict specifying which additional output files to produce (eg strike point loc)

    Returns: Dict with {'status', 'outputs', 'meta_data'}

    """
    logger.info(f'Starting scheduler workflow run for inputs: {machine}, {camera}, {pulse}, {pass_no}')
    logger.debug(f'update_checkpoints={update_checkpoints}, scheduler={scheduler}')
    logger.debug(f'path_user={path_user}, path_calib={path_calib}, path_out={path_out}')

    if analysis_steps is None:
        # TODO: Properly set up top level control of analysis steps
        analysis_steps = dict(camera_shake_correction=False)

    if debug is None:
        debug = {}
    if figures is None:
        figures = {}
    if output_files is None:
        output_files = {}

    if n_cores > 1:
        raise NotImplementedError('Multi-processing has not been implemented yet')

    uda_module, client = uda_utils.get_uda_client(use_mast_client=False, try_alternative=True)

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

    fire_paths = dict(fire_user_dir=path_user, path_uda_output=path_out, calib_dir=path_calib)
    config, config_groups, config_path_fn = user_config.get_user_fire_config(base_paths=fire_paths)
    logger.debug(f'config_groups={config_groups}')

    settings['config'] = config
    fire_paths = config_groups['fire_paths']
    path_figures = fire_paths['figures_dir'].expanduser()

    # Replace users default values from fire_config.json with values passed from command line
    if path_out is not None:
        config['user']['paths']['output']['uda_putdata'] = path_out
    if path_calib is not None:
        config['user']['paths']['calibration_files'] = [path_calib]

    # Load user's default call arguments
    pulse, camera, machine = utils.update_call_args(config['user']['default_params'], pulse, camera, machine)
    interfaces.check_settings_complete(config, machine, camera)

    # Set up/extract useful meta data
    # TODO: Separate camera name and diag_tag?
    camera_info = config['machines'][machine]['cameras'][camera]['device_details']
    diag_tag_raw = camera_info.get('diag_tag_raw', camera)
    diag_tag_analysed = camera_info.get('diag_tag_analysed', uda_utils.get_analysed_diagnostic_tag(diag_tag_raw))

    meta_data.update(dict(pulse=pulse, shot=pulse, camera=camera, machine=machine, pass_no=pass_no, status=1,
                          diag_tag_raw=diag_tag_raw, diag_tag_analysed=diag_tag_analysed,
                          diag_tag_raw_upper=diag_tag_raw.upper(), diag_tag_analysed_upper=diag_tag_analysed.upper(),
                          fire_source_dir=fire_paths['fire_source_dir'], fire_user_dir=fire_paths['fire_user_dir']))
    # Get paths for inserting into path format string (format strings from fire_config.json and elsewhere)
    meta_data.update()
    kwarg_aliases = {'pulse': ['shot'], 'pass_no': ['pass', 'pass_number']}
    meta_data['kwarg_aliases'] = kwarg_aliases
    device_details = config['machines'][machine]['cameras'][diag_tag_raw]['device_details']
    meta_data.update(device_details)
    # Generate id_strings
    meta_data['id_strings'] = interfaces.generate_pulse_id_strings({}, pulse, camera, machine, pass_no)
    meta_data['variables'] = config['variable_meta_data']
    meta_data.update(config['machines'][machine]['cameras'][diag_tag_raw]['device_details'])

    # Load machine plugins
    machine_plugin_paths = config['paths_input']['plugins']['machine']
    machine_plugin_attrs = config['plugins']['machine']['module_attributes']
    machine_plugins, machine_plugins_info = plugins.plugins.get_compatible_plugins(machine_plugin_paths,
                                                                   attributes_required=machine_plugin_attrs['required'],
                                                                   attributes_optional=machine_plugin_attrs['optional'],
                                                                   plugins_required=machine, plugin_type='machine',
                                                                   base_paths=fire_paths)
    machine_plugins, machine_plugins_info = machine_plugins[machine], machine_plugins_info[machine]
    fire.active_machine_plugin = (machine_plugins, machine_plugins_info)
    meta_pulse = plugins.call_plugin_func(machine_plugins, 'pulse_meta_data', args=(pulse,), dummy_ouput={})
    meta_data.update(meta_pulse)

    date = meta_data.get('exp_date', None)
    meta_data['date'] = date

    # Load movie plugins compatible with camera
    # TODO: Move movie reader config information to separate json config file
    movie_plugin_paths = config['paths_input']['plugins']['movie']
    movie_plugin_attrs = config['plugins']['movie']['module_attributes']
    if movie_plugins_filter is None:
        # Try all movie plugins compatible with camera
        movie_plugins_filter = config['machines'][machine]['cameras'][camera]['movie_plugins']

    # Load output format plugins for writing output data
    output_plugin_paths = config['paths_input']['plugins']['output_format']
    output_plugin_attrs = config['plugins']['output_format']['module_attributes']
    output_plugins_active = config['outputs']['active_formats']
    output_variables = config['outputs']['variables']
    output_plugins, output_plugins_info = plugins.plugins.get_compatible_plugins(output_plugin_paths,
                                                                 attributes_required=output_plugin_attrs['required'],
                                                                 attributes_optional=output_plugin_attrs['optional'],
                                                                 plugin_filter=output_plugins_active,
                                                                 plugin_type='output_format',
                                                                 base_paths=fire_paths)

    # Load movie meta data to get lens and integration time etc.
    movie_paths = config['paths_input']['movie_files']
    movie_fns = config['filenames_input']['movie_files']
    movie_reader = plugins.plugins_movie.MovieReader(movie_plugin_paths, plugin_filter=movie_plugins_filter,
                                 movie_paths=movie_paths, movie_fns=movie_fns,
                                 plugin_precedence=None, movie_plugin_definition_file=None, base_paths=fire_paths)
    movie_meta, movie_origin = movie_reader.read_movie_meta_data(pulse=pulse, diag_tag_raw=diag_tag_raw, machine=machine,
                                                         check_output=True, substitute_unknown_values=(not scheduler),
                                                                 meta=meta_data)

    # movie_meta, movie_origin = plugins.plugins_movie.read_movie_meta_data(pulse, camera, machine, movie_plugins,
    #                                                 movie_paths=movie_paths, movie_fns=movie_fns,
    #                                                 substitute_unknown_values=(not scheduler))

    meta_data.update(movie_meta)
    # TODO: format strings to numbers eg lens exposure etc
    lens_focal_length_mm = meta_data.get('lens_in_mm', meta_data['lens'])
    meta_data.update(field_of_view.calc_field_of_view(lens_focal_length_mm, pixel_pitch=meta_data['pixel_pitch'],
                                        image_shape=meta_data['detector_resolution']))
    # NOTE: meta_data['image_shape'] and ipx header hight/width etc prior to calcam image transformations

    # TODO: Tidy up duplicate names for image_resolution

    # Identify and check existence of input files
    paths_input = config['paths_input']['input_files']
    paths_output = {key: Path(path) for key, path in config['paths_output'].items()}
    fn_patterns_input = config['filenames_input']
    fn_pattern_checkpoints = config['checkpoints']['filenames']
    params = {'image_coords': image_coords, **fire_paths, **movie_meta}
    files, lookup_info = interfaces.identify_files(pulse, camera, machine, params=params,
                                        search_paths_inputs=paths_input, fn_patterns_inputs=fn_patterns_input,
                                        paths_output=paths_output, base_paths=fire_paths,
                                        fn_pattern_checkpoints=fn_pattern_checkpoints)
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
            logger.info(f'Producing raycast checkpoint file for first time for camera={diag_tag_raw}, pulse={pulse}: '
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
        io_basic.mkdir(raycast_checkpoint_path_fn, depth=3)
        data_raycast.to_netcdf(raycast_checkpoint_path_fn)
        logger.info(f'Wrote raycast data to: {raycast_checkpoint_path_fn}')
    image_data = xr.merge([image_data, data_raycast])  # , combine_attrs='no_conflicts')


    # TODO: Print out summary of analysis settings prior to analysis

    # TODO: Validate frame range etc

    # Load raw frame data
    movie_data, movie_origin = movie_reader.read_movie_data(pulse=pulse, diag_tag_raw=camera, machine=machine,
                                                            n_start=None, n_end=None, stride=1, transforms=None,
                                                            meta=meta_data)
    (frame_nos, frame_times, frame_data) = movie_data

    logger.info("Movie time period [%g, %g]" % (np.nanmin(frame_times), np.nanmax(frame_times)))

    # TODO: Use techniques in generate_calcam_calib_images to identify bright images for plotting elsewhere
    n_middle = int(np.floor(len(frame_data) / 2))  # Frame number in middle of movie
    frame_ref_pre_transforms = copy(frame_data[n_middle])  # Copy of a frame before calcam transformations applied
    calcam_calib_im_pre_transforms = copy(calcam_calib.get_image(coords='Original'))

    #  window = (Left,Top,Width,Height)
    detector_window_original = movie_meta['detector_window']
    # Get transformed detector window before detector subwindow is applied
    detector_window_display = calcam_calibs.tramsform_detector_window(calcam_calib, detector_window_original,
                                                                      image_coords)
    # update_detector_window MUST be called with ‘original’ detector coords (i.e. before any image rotation, flips etc).
    detector_window_info = calcam_calibs.update_detector_window(calcam_calib, frame_data=frame_data,
                                                                detector_window=detector_window_original, coords='Original')
    calcam_calib_image_windowed = calcam_calib.get_image(coords=image_coords)  # Before detector_window applied

    # Apply transformations (rotate, flip etc.) to get images "right way up" if requested.
    # Must be applied after detector_window
    meta_data['image_shape_original'] = meta_data['image_shape']
    frame_data = calcam_calibs.apply_frame_display_transformations(frame_data, calcam_calib, image_coords)
    image_shape = frame_data.shape[-2:]
    meta_data['image_shape'] = image_shape  # Current shape of the data
    if image_coords.capitalize() == 'Display':
        meta_data['image_shape_display'] = image_shape
        meta_data['detector_window_display'] = detector_window_display

    frame_data = data_structures.movie_data_to_dataarray(frame_data, frame_times, frame_nos,
                                                         meta_data=meta_data)
    image_data = xr.merge([image_data, frame_data])  # , combine_attrs='no_conflicts')

    image_shape = np.array(frame_data.shape[1:])  # NOTE: meta_data['image_shape'] and ipx header info is without image
    # transformations

    if np.any(np.diff(frame_data['t']) < 0):
        t_out_of_order = (frame_data["t"].diff(dim='n')) <= 0
        n_out_of_order = frame_data['n'].where(t_out_of_order, drop=True)
        if (len(n_out_of_order) == 1) and (n_out_of_order[0] == frame_data['n'][-1]):
            logger.warning(f'Bad timestamp at end of movie: {np.array(image_data["t"][-4:])}. Dropping it.')
            image_data = image_data.drop_sel(dict(n=n_out_of_order))
        else:
            logger.warning(f'Movie data contains non-monotonic time data. Re-ordering frames by timestamps. '
                           f'{t_out_of_order}')
            image_data = image_data.sortby('t')
        frame_data = image_data['frame_data']

    if (debug.get('movie_intensity_stats-raw', False)):
        # Force plot if there are any saturated frames
        temporal_figures.plot_movie_intensity_stats(frame_data, meta_data=meta_data, num='movie_intensity_stats-raw')

    # if (movie_origin['plugin'] == 'ipx') and (machine == 'mast_u') and (np.max(frame_data) <= 2**12):
    #     frame_data *= 4  # TODO: Remove when ipx writing fixed!
    #     logger.warning('Multiplied data from MAST-U ipx file by factor of 4 due to mastvideo ipx writing issue')

    clock_info = fire.plugins.plugins.call_plugin_func(machine_plugins, 'get_camera_external_clock_info',
                                                       args=(camera, pulse))

    # TODO: move time checks to separate function
    if clock_info is not None:
        if (not np.isclose(np.abs(movie_meta.get('trigger', movie_meta.get('t_before_pulse'))),
                           np.abs(clock_info['clock_t_window'][0]))):
            raise ValueError(f'Movie and camera clock start times do not match. '
                             f'movie={movie_meta["t_before_pulse"]}, clock={clock_info["clock_t_window"]}')
        if np.abs(movie_meta['fps'] - clock_info['clock_frequency']) > 1:
            message = (f'Movie and camera clock frequencies do not match. '
                             f'movie={movie_meta["fps"]}, clock={clock_info["clock_frequency"]}')
            logger.warning(message)
            pass
            apply_time_axis_correction = False
            if apply_time_axis_correction:
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

    # Perform bad pixel replacement (BPR)
    mask_bad_pixels = movie_meta.get('bad_pixel_frame')
    frame0_pre_bpr = copy(frame_data[0])
    frame_mid_pre_bpr = copy(frame_data[n_middle])
    if mask_bad_pixels is None:
        # TODO: Lookup known anomalies to mask - eg stray LP cable, jammed shutter - replace image data with nans?

        # TODO: Detect anommalies
        # TODO: Monitor number of bad pixels for detector health - option to separate odd/even frames for FLIR sensors
        # detect_bpr = True
        detect_bpr = False
        if detect_bpr:
            # bad_pixels, frame_no_dead = image_processing.find_outlier_pixels(frame_raw, n_sigma_tol=10, check_edges=True)
            bad_pixels = image_processing.identify_bad_pixels(frame_mid_pre_bpr, method='blur_diff',  # 'blur_diff', 'flicker_offset
                                                n_sigma_tol=None, n_bad_pixels_expected=678*1e1, blur_axis=1)
            (bad_pixels, mask_bad_pixels, threshold_hot, detector_bands, images_blured) = bad_pixels
        elif (diag_tag_raw == 'rit'):
            # TODO: Move to file lookup with calcam calibration etc
            path_fn_bpr = '/home/tfarley/repos/air/fire/input_files/mast_u/badPixel_231864.BPR'
            bad_pixels = io_basic.read_csv(path_fn_bpr, names=('y_pix', 'x_pix'))
            mask_bad_pixels = image_processing.bpr_list_to_mask(bad_pixels, detector_window_display)
            logger.info(f'Read bad pixel coordinate list from file: {path_fn_bpr}')

    remove_bad_pixels = False
    # remove_bad_pixels = True
    if remove_bad_pixels and (mask_bad_pixels is not None):
        # TODO: Else read BPR from file?
        image_data['frame_data'][:] = image_processing.apply_bpr_correction(frame_data, mask_bad_pixels,
                                                                         method='median_loop', kwargs=dict(size=5))
        frame_data = image_data['frame_data']
    else:
        logger.info('NOT applying bad pixel replacement (BPR)')

    if (debug.get('bad_pixels', False)):  # TODO: Remove duplicate
        debug_plots.plot_bad_pixels(mask_bad_pixels=mask_bad_pixels, frame_data=frame_mid_pre_bpr,
                                    frame_data_corrected=frame_data)
        temporal_figures.plot_movie_intensity_stats(frame_data, meta_data=meta_data, num='movie_intensity_stats-bpr')

    # Detect saturated pixels, uniform frames etc
    bad_frames_info = data_quality.identify_bad_frames(frame_data, bit_depth=movie_meta['bit_depth'],
                                                       n_discontinuities_expected=1e-3, meta_data=meta_data, scheduler=scheduler,
                                                       n_sigma_multiplier_high=1.5,  #  2.5, 4
                                                       n_sigma_multiplier_low=1.5,
                                                       n_sigma_multiplier_start=1,
                                                       raise_on_saturated=False, debug_plot=debug.get('bad_frames_intensity', False))
    frames_nos_discontinuous = bad_frames_info['discontinuous']['frames']['n']
    frame_data_fixed, modified_frames = data_quality.remove_bad_frames(frame_data, frames_nos_discontinuous,
                            remove_opening_closing=True, interpolate_middle=True, nan_middle=False, meta_data=meta_data,
                            debug_plot=debug.get('bad_frames_images', False), scheduler=scheduler)
    # Merge to set bad frames to interpolated/nan values.
    # Note: Merging subset DataArray with join and compat keywords DOES reduce coord ranges, so don't use .drop_sel()
    image_data = xr.merge([image_data, frame_data_fixed], join='right', compat='override')
    # Merging subset DataArray doesn't reduce coord ranges so need to drop bad opening or closing coords
    # image_data = image_data.drop_sel(dict(n=modified_frames['removed']))
    frame_data = image_data['frame_data']

    # TODO: Detect frames with non-uniform time differences (see heat flux func)


    # Use dark parts of image to detect dark level drift
    # apply_dark_level_correction = True
    apply_dark_level_correction = False

    if apply_dark_level_correction or debug.get('dark_level', False):
        # TODO: Store dark level correction and mask in dataset
        dark_level, dark_level_correction_factors, mask_dark = camera_checks.get_dark_level_drift(image_data, plot=False)

    if apply_dark_level_correction:
        frame_data = camera_checks.correct_dark_level_drift(image_data['frame_data'], dark_level_correction_factors)
        image_data['frame_data'] = frame_data
    else:
        logger.debug('NOT applying dark level drift correction')

    if (debug.get('dark_level', False)):
        camera_checks.plot_dark_level_variation(mask_dark=mask_dark, frame_data=image_data['frame_data'],
                                                dark_level=dark_level)

    t = image_data['t']

    if (debug.get('movie_intensity_stats-corrected', False) or
            ((not scheduler) and (bad_frames_info['saturated']['n_bad_frames']))):
        # Force plot if there are any saturated frames
        fig, ax, ax_n = temporal_figures.plot_temporal_stats(image_data['frame_data'], meta_data=meta_data,
                                                             num='movie_intensity_stats-corrected',
                                                             bit_depth=meta_data['bit_depth'], show=False)
        if modified_frames:
            # Label ends of movie that are discarded due discontinuous intensities etc
            ax_n.axvline(x=modified_frames['n_removed_start'], ls=':', color='k', label='Clipped bad frames from start')
            ax_n.axvline(x=len(image_data['frame_data']) - 1 - modified_frames['n_removed_end'], ls=':', color='k',
                         label='Clipped bad frames from end')
            for n in modified_frames['corrected']:
                ax_n.axvline(x=n, ls=':', lw=1, color='r', label='Interpolated bad frame')
        plot_tools.show_if(show=True)

    # TODO: Add coordinates for pre-transformation frame data
    # image_data_ref = xr.Dataset(coords=image_data.coords)
    # image_data_ref.coords['n_ref'] = ('n_ref', [n_middle])
    # image_data_ref['frame_ref_pre_transforms'] = (('n_ref', 'y_pix_raw', 'x_pix_raw'),
    #                                               frame_ref_pre_transforms[np.newaxis, ...])

    # TODO: Lookup and apply t_offset correction to frame times? See IDL get_toffset.pro reading toffset.dat

    # Fix camera shake
    if analysis_steps['camera_shake_correction']:
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
        # TODO: Try using calcam.movement.detect_movement(ref, moved) instead of own functions
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

    # TODO: Detect 'twinkling pixels' by looking for pixels with abnormally large ptp variation for steady state images
    # TODO: Detect missing frames in time stamps, regularity of frame rate
    # TODO: Compare sensor temperature change during/between pulses to monitor sensor health, predict frame losses etc.

    # TODO: Rescale DLs to account for window transmission - Move here out of dl_to_temerature()?

    # Apply NUC correction
    # nuc_frame = get_nuc_frame(origin='first_frame', frame_data=frame_data)
    # TODO: Consider using min over whole (good frame) movie range to avoid negative values from nuc subtraction
    # TODO: consider using custon nuc time/frame number range for each movie? IDL sched had NUC time option
    # TODO: Look up first shot of day and load starting nuc frames from there to avoid warming from preceding shots
    n_digitisers = 1  # FLIR SC7500 has dual digitisers
    nuc_frames = nuc.get_nuc_frames(origin={'n': nuc_frame_range}, frame_data=image_data['frame_data'],
                                              reduce_func='mean', n_digitisers=1)

    # nuc_frame = get_nuc_frame(origin={'n': [None, None]}, frame_data=frame_data, reduce_func='min')
    frame_data_nuc = nuc.apply_nuc_correction(image_data['frame_data'], nuc_frames, raise_on_negatives=False)
    frame_data_nuc = data_structures.attach_standard_meta_attrs(frame_data_nuc, varname='frame_data_nuc',
                                                                key='frame_data')
    image_data['nuc_frames'] = nuc_frames
    image_data['frame_data_nuc'] = frame_data_nuc

    if (debug.get('movie_intensity_stats-nuc', False) or
            ((not scheduler) and (bad_frames_info['saturated']['n_bad_frames']))):
        # Force plot if there are any saturated frames
        fig, ax, ax_n = temporal_figures.plot_temporal_stats(image_data['frame_data_nuc'], meta_data=meta_data,
                                                             num='movie_intensity_stats-nuc', show=False)
        if modified_frames:
            # Label ends of movie that are discarded due discontinuous intensities etc
            ax_n.axvline(x=modified_frames['n_removed_start'], ls=':', color='k', label='Clipped bad frames from start')
            ax_n.axvline(x=len(image_data['frame_data']) - 1 - modified_frames['n_removed_end'], ls=':', color='k',
                         label='Clipped bad frames from end')
            for n in modified_frames['corrected']:
                ax_n.axvline(x=n, ls=':', lw=1, color='r', label='Interpolated bad frame')
        plot_tools.show_if(show=True)

    if output_files.get('raw_frame_image', False):
        # nuc_out = True
        nuc_out = False
        n_ref = 240
        path_fn = paths_output['raw_images'] / f'{machine}-{diag_tag_raw}-{pulse}-n{n_ref}{"_nuc"*nuc_out}.png'
        key = 'frame_data_nuc' if nuc_out else 'frame_data'
        image_figures.figure_frame_data(image_data, n=n_ref, key=key, label_outliers=False,
                                            axes_off=True, show=True, save_fn_image=path_fn)

    if debug.get('debug_detector_window', False):  # Need to plot before detector window applied to calibration
        debug_plots.debug_detector_window(detector_window=detector_window_original, frame_data=image_data,
                                          calcam_calib=calcam_calib,  # image_full_frame=calcam_calib_image_full_frame,
                                          image_coords=image_coords, meta_data=meta_data)
    if debug.get('movie_data_animation', False):
        save_path_fn = None
        # frame_range = [40, 410]
        frame_range = [40, None]
        cbar_range = [0, 99.9]  # percentage of range
        if not None in frame_range:
            frame_range = np.clip(frame_range, *meta_data['frame_range'])

        image_figures.animate_frame_data(image_data, key='frame_data', nth_frame=1, duration=15,
                                         n_start=frame_range[0], n_end=frame_range[1], save_path_fn=save_path_fn,
                                         cbar_range=cbar_range,
                                         frame_label=f'{camera.upper()} {pulse} $t=${{t:0.1f}} ms',
                                         cbar_label='$DL_{raw}$ [DL]',
                                         label_values={'t': image_data['t'].values * 1e3}, show=True)

    if debug.get('movie_data_nuc_animation', False):
        save_path_fn = None
        # frame_range = [40, 410]
        frame_range = [440, 470]  # RIR 29541
        # frame_range = [40, None]
        cbar_range = [0, 99.9]  # percentage of range
        if not None in frame_range:
            frame_range = np.clip(frame_range, *meta_data['frame_range'])

        image_figures.animate_frame_data(image_data, key='frame_data_nuc', nth_frame=1, duration=15,
                                         n_start=frame_range[0], n_end=frame_range[1], save_path_fn=save_path_fn,
                                         cbar_range=cbar_range,
                                         frame_label=f'{camera.upper()} {pulse} $t=${{t:0.1f}} ms',
                                         cbar_label='$DL-DL_{NUC}$ [DL]',
                                         label_values={'t': image_data['t'].values * 1e3}, show=True)

    if debug.get('movie_data_nuc', False):
        debug_plots.debug_movie_data(image_data, key='frame_data_nuc')

    if debug.get('specific_frames', False):
        n_check = 452  # 218
        debug_plots.debug_movie_data(image_data, frame_nos=np.arange(n_check, n_check+4), key='frame_data_nuc')
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
        fn_spatial_res = path_figures / f'spatial_res_{pulse}_{diag_tag_raw}_{spatial_res_type}.png'
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

    message = f'Camera calibration contains {calcam_calib.n_subviews} sub-views: {calcam_calib.subview_names}'
    logger.info(message) if (calcam_calib.n_subviews > 1) else logger.debug(message)

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
        # save_path_fn = paths_output['gifs'] / f'{machine}-{diag_tag_raw}-{pulse}_temperature_movie.gif'
        # save_path_fn = path_figures / f'rit_{pulse}_temperature_movie.gif'
        save_path_fn = None
        frame_range = [40, None]
        # frame_range = [40, 410]
        cbar_range = [0, 99.9]  # percentage of range
        # cbar_range = None
        if not None in frame_range:
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
        # TODO: Create separate dataset for each path to keep coord name simple, then combine into single dataset
        # with complicated coord names at end ie Dataset.rename()
        # TODO: Create functions to combine and split datasets by path
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

        if equilibrium:
            # TODO: Pass efit converged times
            times = np.array(frame_data['t'])
            efit_utils.extract_eqm_info_for_path(pulse, path_data, times=times, path_name=analysis_path_key)
            efit_data = efit_utils.get_efit_data(shot=pulse, calc_bfield=True)
            t = 0

            ['Br', 'Bz', 'Bphi', 'Bpol', 'Btot']
            'r', 'z', 't', 'q95', 'psiN', 'lower_xpoint_r', 'lower_xpoint_z', 'lower_xpoint_psin'

        # Sort analysis path in order of increasing R to avoid reversing sections of profiles!
        path_data = path_data.sortby(f'R_in_frame_path{i_path}', ascending=True)  # TODO: Detect non-mono and tidy into func?
        path_data[f'i_in_frame_path{i_path}'] = np.sort(np.array(path_data[f'i_in_frame_path{i_path}']))

        path_data, path_data_non_mono = data_quality.filter_non_monotonic(path_data, coord=f'R_in_frame_path{i_path}')
        path_data, path_data_non_mono = data_quality.filter_non_monotonic(path_data,
                                        coord=f's_global_in_frame_path{i_path}', non_monotonic_prev=path_data_non_mono)
        # TODO: Understand missing ~3 values at end of path
        n_non_monotonic = len(path_data_non_mono)
        if n_non_monotonic > 0:
            logger.warning(f'Removed {n_non_monotonic} non-monotonic elements from analysis path {i_path}')
            logger.debug(f'Path coordinates of non monotonic path elements: {path_data_non_mono.coords}')

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
        logger.debug(f'Surfaces visible along path "{analysis_path_name}": {visible_surfaces_path}')

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
                                              num='temperature_vs_R_t', robust=False, meta=meta_data, t_range=None,
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

        if debug.get('heat_flux_vs_R_t-raw', False):
            # robust = True
            robust = False
            extend = None
            debug_plots.debug_plot_profile_2d(path_data, param='heat_flux', path_names=analysis_path_key,
                                              robust=robust, extend=extend, meta=meta_data, mark_peak=True,
                                              machine_plugins=machine_plugins, t_range=None)
        if figures.get('heat_flux_vs_R_t-robust', False):
            robust = True
            extend = 'both'
            fn = f'heat_flux_vs_R_t-robust-{machine}_{diag_tag_analysed}_{pulse}.png'
            save_path_fn = path_figures / 'heat_flux_vs_R_t-robust'/ diag_tag_analysed / fn
            debug_plots.debug_plot_profile_2d(path_data, param='heat_flux', path_names=analysis_path_key,
                                              extend=extend, robust=robust, meta=meta_data, t_range=None,
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
            heat_flux = path_data[f'heat_flux_amplitude_global_peak_{path}'].values
            heat_flux_thresh = np.nanmin(heat_flux) + 0.03 * (np.nanmax(heat_flux) - np.nanargmin(heat_flux))
            debug_plots.debug_plot_temporal_profile_1d(path_data_all, params=('heat_flux_R_peak', 'heat_flux_amplitude_global_peak'),
                                                       path_name=analysis_path_key, x_var='t',
                                                       heat_flux_thresh=heat_flux_thresh, meta_data=meta_data)
        if output_files.get('strike_point_loc', False):
            fn = f'strike_point_loc-{machine}-{diag_tag_analysed}-{pulse}-{analysis_path_name}.csv'
            path_fn = Path(str(paths_output['csv_data']).format(**fire_paths)) / diag_tag_analysed / fn
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

    outputs = plugins.plugins_output_format.write_to_output_format(output_plugins, path_names=analysis_path_keys,
                    path_data=path_data_all, image_data=image_data,
                    variable_names_path=output_variables['analysis_path'], variable_names_time=output_variables['time'],
                    variable_names_image=output_variables['image'],
                    device_info=device_details, header_info=output_header_info, meta_data=meta_data,
                    raise_on_fail=raise_on_fail,
                    client=client, verbose=True)
    # write_processed_ir_to_netcdf(data, path_fn_out)

    archive_netcdf_output = True
    # archive_netcdf_output = False
    if archive_netcdf_output and not scheduler:
        path_fn_netcdf = outputs.get('uda_putdata', {}).get('path_fn')
        interfaces.archive_netcdf_output(path_fn_netcdf, meta_data=meta_data)

    logger.info(f'Finished scheduler workflow')

    status = 'success'
    return {'status': status, 'outputs': outputs, 'meta_data': meta_data}

def copy_output(outputs, clean_netcdf=True, copy_to_uda_scrach=True):
    path_fn_netcdf = outputs.get('uda_putdata', {}).get('path_fn')

    shot = outputs['meta_data']['shot']

    if copy_to_uda_scrach and (path_fn_netcdf is not None):
        path_fn_projects = Path(f'/projects/SOL/Data_analysis/IR/ait/{shot}') / path_fn_netcdf.name
        path_fn_scratch = Path('/common/uda-scratch/IR') / path_fn_netcdf.name
        path_fn_scratch.parent.mkdir(exist_ok=True)
        path_fn_projects.parent.mkdir(exist_ok=True)
        path_fn_scratch.write_bytes(path_fn_netcdf.read_bytes())
        path_fn_projects.write_bytes(path_fn_netcdf.read_bytes())
        logger.info(f'Copied uda output file to: {str(path_fn_scratch)}, {str(path_fn_projects)}')

    if clean_netcdf:
        if ('uda_putdata' in outputs) and (outputs['uda_putdata']['success']):

            if path_fn_netcdf.is_file():
                io_basic.copy_file(path_fn_netcdf, path_fn_scratch)

                path_fn_netcdf.unlink()
                logger.info(f'Deleted uda output file to avoid clutter since run from main in scheduler_workflow.py: '
                            f'{path_fn_netcdf}')



