#!/usr/bin/env python

"""


Created: 
"""

import logging
from pathlib import Path

import numpy as np

import calcam

import fire
from fire import fire_paths
from fire.interfaces import interfaces
from fire.plugins import plugins
from fire.plotting import debug_plots, image_figures, spatial_figures, temporal_figures
from fire.scripts.read_pickled_ir_data import read_data_for_pulses_pickle

logger = logging.getLogger(__name__)
logger.propagate = False


def review_analysed_shot_pickle(pulse, camera='rit', machine='mast_u', debug_figures=None, recompute=False):
    logger.info(f'Reviewing {machine}, {camera}, {pulse}')
    print(f'Reviewing {machine}, {camera}, {pulse}')

    data, data_unpacked = read_data_for_pulses_pickle(camera=camera, pulses=pulse, machine=machine,
                                                      generate=True, recompute=recompute)[pulse]

    image_data = data['image_data']
    path_data = data['path_data']

    meta = data['meta_data']
    # meta = dict(pulse=pulse, camera=camera, machine=machine)

    review_analysed_shot(image_data, path_data, meta=meta, debug=debug_figures)

def review_analysed_shot(image_data, path_data, meta, debug=None, output=None):
    if debug is None:
        debug = {}
    if output is None:
        output = {}


    meta_data = meta

    # Required meta data
    pulse = meta['pulse']
    camera = meta['camera']
    machine = meta['machine']
    files = meta['files']
    analysis_path_names = meta['analysis_path_names']
    analysis_path_keys = meta['analysis_path_keys']
    analysis_path_labels = meta.get('analysis_path_labels', analysis_path_names)

    # Optional data
    frame_data = image_data.get('frame_data')

    calcam_calib = calcam.Calibration(load_filename=str(files['calcam_calib']))

    config = interfaces.json_load(fire_paths['config'], key_paths_drop=('README',))

    paths_input = config['paths_input']['input_files']
    paths_output = {key: Path(path) for key, path in config['paths_output'].items()}

    # Load machine plugins
    machine_plugin_paths = config['paths_input']['plugins']['machine']
    machine_plugin_attrs = config['plugins']['machine']['module_attributes']
    machine_plugins, machine_plugins_info = plugins.get_compatible_plugins(machine_plugin_paths,
                                                                                   attributes_required=
                                                                                   machine_plugin_attrs['required'],
                                                                                   attributes_optional=
                                                                                   machine_plugin_attrs['optional'],
                                                                                   plugins_required=machine,
                                                                                   plugin_type='machine')
    machine_plugins, machine_plugins_info = machine_plugins[machine], machine_plugins_info[machine]
    fire.active_machine_plugin = (machine_plugins, machine_plugins_info)



    n_middle = int(np.floor(len(frame_data)/2))  # Frame number in middle of movie

    if debug.get('calcam_calib_image', False):
        frame_ref_pre_transforms = None
        debug_plots.debug_calcam_calib_image(calcam_calib, frame_data=frame_data,
                                             frame_ref=frame_ref_pre_transforms, n_frame_ref=n_middle)

    # if debug.get('camera_shake', False):
    #     debug_plots.debug_camera_shake(pixel_displacements=pixel_displacemnts, n_shake_ref=n_shake_ref)

    # if debug.get('debug_detector_window', False):  # Need to plot before detector window applied to calibration
    #     debug_plots.debug_detector_window(detector_window=detector_window, frame_data=image_data,
    #                                       calcam_calib=calcam_calib, image_full_frame=calcam_calib_image_full_frame,
    #                                       image_coords=image_coords)

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
        debug_plots.debug_movie_data(image_data, frame_nos=np.arange(n_check, n_check+4), key='frame_data') #
        # frame_data_nuc

    if (debug.get('movie_intensity_stats', False)):
        # Force plot if there are any saturated frames
        temporal_figures.plot_movie_intensity_stats(frame_data, meta_data=meta_data)

    if debug.get('spatial_res', False):
        debug_plots.debug_spatial_res(image_data)

    # if figures.get('spatial_res', False):
    #     fn_spatial_res = path_figures / f'spatial_res_{pulse}_{camera}.png'
    #     image_figures.figure_spatial_res_max(image_data, clip_range=[None, 20], save_fn=fn_spatial_res, show=True)

    if False and debug.get('spatial_coords', False):
        phi_offset = 112.3
        # points_rzphi = [[0.80, -1.825, phi_offset-42.5], [1.55, -1.825, phi_offset-40]]  # R, z, phi - old IDL
        points_rzphi = None  # R, z, phi
        # analysis path
        # points_rzphi = [[0.80, -1.825, -42.5]]  # R, z, phi - old IDL analysis path
        # points_pix = np.array([[256-137, 320-129], [256-143, 320-0]]) -1.5  # x_pix, y_pix - old IDL analysis path
        points_pix = np.array([[256-133, 320-256], [256-150, 320-41]]) -1.5  # x_pix, y_pix - old IDL analysis path: [256, 133], [41, 150]
        # points_pix = None
        # points_pix plotted as green crosses
        debug_plots.debug_spatial_coords(image_data, points_rzphi=points_rzphi, points_pix=points_pix)

    if debug.get('surfaces', False):
        debug_plots.debug_surfaces(image_data)

    if debug.get('temperature_im', False):
        debug_plots.debug_temperature_image(image_data)

    if debug.get('movie_temperature_animation', False):
        # save_path_fn = paths_output['gifs'] / f'{machine}-{camera}-{pulse}_temperature_movie.gif'
        save_path_fn = None
        cbar_range = [0, 99.8]  # percentile of range
        # cbar_range = [0, 99.9]  # percentile of range
        # cbar_range = [0, 100]  # percentile of range
        # cbar_range = None
        frame_range = [40, 250]
        image_figures.animate_frame_data(image_data, key='temperature_im', nth_frame=1, n_start=frame_range[0],
                                         n_end=frame_range[1], save_path_fn=save_path_fn, cbar_range=cbar_range,
                                         frame_label=f'{camera.upper()} {pulse} $t=${{t:0.1f}} ms',
                                         cbar_label='$T$ [$^\circ$C]',
                                         label_values={'t': meta_data['frame_times']*1e3})

    for i_path, (analysis_path_key, analysis_path_name) in enumerate(zip(analysis_path_keys, analysis_path_names)):
        path = analysis_path_key
        meta_data['path_label'] = analysis_path_labels[i_path]

        if debug.get('poloidal_cross_sec', False):
            spatial_figures.figure_poloidal_cross_section(image_data=image_data, path_data=path_data, pulse=pulse, no_cal=True,
                                                                        show=True)
        if debug.get('spatial_coords', False):
            debug_plots.debug_spatial_coords(image_data, path_data=path_data, path_name=analysis_path_key)

        if debug.get('temperature_vs_R_t', False):
            debug_plots.debug_plot_profile_2d(path_data, param='temperature', path_names=analysis_path_key,
                                              robust=False, meta=meta_data, machine_plugins=machine_plugins,
                                              verbose=True)

        if debug.get('heat_flux_vs_R_t-robust', False):
            robust = True
            # robust = (2, 100)
            # robust = False
            extend = 'both'
            # extend = 'min'
            # extend = 'neither'
            debug_plots.debug_plot_profile_2d(path_data, param='heat_flux', path_names=analysis_path_key, extend=extend,
                                              robust=robust, meta=meta_data, machine_plugins=machine_plugins)

        if debug.get('heat_flux_vs_R_t-raw', False):
            robust = False
            debug_plots.debug_plot_profile_2d(path_data, param='heat_flux', path_names=analysis_path_key,
                                              extend='neither', robust=robust,
                                              meta=meta_data, machine_plugins=machine_plugins)

        if debug.get('analysis_path', False):
            # TODO: Finish  debug_analysis_path_2d
            # debug_plots.debug_analysis_path_2d(image_data, path_data, path_names=analysis_path_key,
            #                        image_data_in_cross_sections=True, machine_plugins=machine_plugins)
            debug_plots.debug_analysis_path_1d(image_data, path_data, path_names=analysis_path_key,
                           image_data_in_cross_sections=True, machine_plugins=machine_plugins,
                           pupil_coords=meta.get('calcam_pupilpos'),
                           keys_profiles=(
                           ('frame_data_min(i)_{path}', 'frame_data_mean(i)_{path}', 'frame_data_max(i)_{path}'),
                           ('temperature_min(i)_{path}', 'temperature_mean(i)_{path}', 'temperature_max(i)_{path}'),
                           ('heat_flux_min(i)_{path}', 'heat_flux_mean(i)_{path}', 'heat_flux_max(i)_{path}'),
                           ('s_global_{path}', 'R_{path}'),
                           ('spatial_res_x_{path}', 'spatial_res_y_{path}', 'spatial_res_linear_{path}')
                           ))

        if debug.get('timings', False):
            debug_plots.debug_plot_timings(path_data, pulse=pulse)

        if debug.get('strike_point_loc', False):
            heat_flux = path_data[f'heat_flux_peak_{path}'].values
            heat_flux_thresh = np.nanmin(heat_flux) + 0.03 * (np.nanmax(heat_flux)-np.nanargmin(heat_flux))
            debug_plots.debug_plot_temporal_profile_1d(path_data, params=('heat_flux_r_peak', 'heat_flux_peak'),
                                                       path_name=analysis_path_keys, x_var='t',
                                                       heat_flux_thresh=heat_flux_thresh, meta_data=meta_data)
        if output.get('strike_point_loc', False):
            path_fn = Path(paths_output['csv_data']) / f'strike_point_loc-{machine}-{camera}-{pulse}.csv'
            interfaces.to_csv(path_fn, path_data, cols=f'heat_flux_r_peak_{path}', index='t', x_range=[0, 0.6],
                              drop_other_coords=True, verbose=True)


if __name__ == '__main__':
    import pyuda
    client = pyuda.Client()

    # pulse = 43183  # Nice strike point sweep to T5, but negative heat fluxes
    # pulse = 43177
    # pulse = 43530
    # pulse = 43534
    # pulse = 43547
    # pulse = 43415  # Peter Ryan's strike point sweep for LP checks

    # pulse = 43583  # 2xNBI
    # pulse = 43587  #
    # pulse = 43591  #
    # pulse = 43596  #
    # pulse = 43610  #
    # pulse = 43620  #
    # pulse = 43624  #
    # pulse = 43662  #

    # pulse = 43624  #
    # pulse = 43648  #

    # pulse = 43611
    # pulse = 43613
    pulse = 43614

    debug = {'calcam_calib_image': False, 'debug_detector_window': False,
             'movie_intensity_stats': False,
         'movie_data_animation': False, 'movie_data_nuc_animation': False, 'movie_temperature_animation': False,
         'spatial_coords': False,
         'spatial_res': False,
         'movie_data_nuc': False, 'specific_frames': False, 'camera_shake': False, 'temperature_im': False,
         'surfaces': False,
         'analysis_path': True,
         'temperature_vs_R_t': False,
         'heat_flux_vs_R_t-robust': True, 'heat_flux_vs_R_t-raw': True,
         'timings': True, 'strike_point_loc': True,
         # 'heat_flux_path_1d': True,
         }
    # debug = {k: True for k in debug}
    # debug = {k: False for k in debug}

    # recompute = True
    recompute = False

    review_analysed_shot_pickle(pulse=pulse, debug_figures=debug, recompute=recompute)
    pass

