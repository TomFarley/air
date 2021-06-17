#!/usr/bin/env python

"""


Created: 
"""

import logging
from pathlib import Path
from typing import Union, Sequence

import numpy as np
import matplotlib.pyplot as plt

import calcam

import fire
from fire import fire_paths
from fire.interfaces import interfaces
from fire.misc.utils import make_iterable
from fire.plugins import plugins
from fire.plugins.output_format_plugins.pickle_output import read_output_file
from fire.misc.utils import make_iterable
from fire.plotting import plot_tools, debug_plots, spatial_figures
from fire.interfaces import uda_utils
from fire.scripts.read_pickled_ir_data import read_data_for_pulses_pickle

logger = logging.getLogger(__name__)
logger.propagate = False


def compare_shots_1d(camera: str, signals: dict, pulses: dict, machine:str= 'mast_u', ax=None, show=True):

    data = read_data_for_pulses_pickle(camera, pulses, machine)

    for signal, signal_props in signals.items():
        fig, ax_i, ax_passed = plot_tools.get_fig_ax(ax=ax)
        for pulse, pulse_props in pulses.items():
            path_data = data[pulse][0]['path_data']
            if signal in path_data:
                kwargs = dict()
                kwargs.update(pulse_props)
                kwargs.update(signal_props)

                plot_kwargs = dict(alpha=0.7)

                spatial_figures.figure_spatial_profile_1d(path_data, key=signal, ax=ax_i,
                                                          label=f'{pulse}', plot_kwargs=plot_kwargs, show=False,
                                                          **kwargs)
            else:
                raise NotImplementedError(signal)
        plot_tools.annotate_axis(ax_i, f'{signal}')

        plot_tools.legend(ax=ax_i, legend=True, only_multiple_artists=False)
        plot_tools.save_fig(path_fn=None, fig=fig)
        plot_tools.show_if(show=show)

def compare_shots_1d_with_uda_signals(camera: str, signals_ir: dict, signals_uda: dict, pulses: dict, machine:str= 'mast_u',
                                      show=True):
    fig, axes, ax_passed = plot_tools.get_fig_ax(ax_grid_dims=(2, 1), sharex=True, axes_flatten=True)

    if isinstance(pulses, (tuple, list, np.ndarray)):
        pulses = {pulse: {} for pulse in pulses}

    ax = axes[0]
    compare_shots_1d(camera, signals_ir, pulses=pulses, machine=machine, ax=ax, show=False)

    ax = axes[1]
    for signal in make_iterable(signals_uda):
        for pulse in pulses.keys():
            uda_utils.plot_uda_signal(signal, pulse, ax=ax, alpha=0.7, show=False)

    plot_tools.show_if(show=show)


def compare_shots_2d(camera: str, signals: dict, pulses: Union[dict, Sequence], machine:str= 'mast_u',
                     t_range=None, r_range=None, t_wins=None, set_ax_lims_with_ranges=True, robust=True,
                     machine_plugins=None,show=True, colorbar_kwargs=None, **kwargs):
    pulses = make_iterable(pulses)
    data = read_data_for_pulses_pickle(camera, pulses, machine)

    colorbar_kws = dict(position='top', size='3%')
    if colorbar_kwargs is not None:
        colorbar_kws.update(colorbar_kwargs)

    fig, axes, ax_passed = plot_tools.get_fig_ax(ax_grid_dims=(1, len(make_iterable(pulses))), axes_flatten=True,
                                                 figsize=(14, 8), sharex=True, sharey=True)
    axes = make_iterable(axes)

    path_key = 'path0'  # TODO: Generalise to other paths!
    for signal, signal_props in signals.items():
        for i_ax, (ax, pulse) in enumerate(zip(axes, pulses)):
            if isinstance(pulse, (tuple, list)):  # When pulse is passed in tuple with dict of kwargs for that pulse
                pulse, kws = pulse
            data_paths = data[pulse][0]['path_data']
            meta = data[pulse][0]['meta_data']

            # meta = dict(pulse=pulse, camera=camera, machine=machine)
            meta['path_label'] = meta['analysis_path_labels'][meta['analysis_path_keys'].index(path_key)]
            debug_plots.debug_plot_profile_2d(data_paths, param=signal, ax=ax, path_names=path_key, robust=robust,
                                              annotate=True, meta=meta, t_wins=t_wins, t_range=t_range, r_range=r_range,
                                              machine_plugins=machine_plugins, show=False,
                                              colorbar_kwargs=colorbar_kws,
                                              set_data_coord_lims_with_ranges=set_ax_lims_with_ranges, **kwargs)
            if i_ax > 0:
                ax.yaxis.label.set_visible(False)
    plt.tight_layout()
    plot_tools.show_if(show, tight_layout=False)
    plot_tools.save_fig(None)

if __name__ == '__main__':
    machine = 'mast_u'
    camera = 'rit'

    # r_t2_bounds = [0.540, 0.905]
    # r_t5_bounds = [1.395, 1.745]
    r_t2_bounds = [0.703, 0.90]
    r_t5_bounds = [1.4, 1.75]
    # x_range = (0.7, 0.9)  # for R on x axis, T2
    # x_range = (1.4, 1.75)  # for R on x axis, T5

    # calcam_calib = calcam.Calibration(load_filename=str(files['calcam_calib']))

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

    # signals = {'heat_flux_path0': dict(x_coord='s_global_path0', reduce=[('t', np.mean, ())], offset_x_peak=True)}
    signals = {'heat_flux_peak_path0': dict()}
    # pulses = {43163: {}, 43183: {}}
    # pulses = {43413: {}, 43415: {}}
    # pulses = {43583 : {}}
    # signals_uda = ['ip', 'Da']
    # signals_uda = ['/xim/da/hm10/t']  # , 'AIM_DA/HM10/T']
    signals_uda = ['/AMC/ROGEXT/TF']  # , 'AIM_DA/HM10/T']
    signals_uda = ['/AMC/ROGEXT/TF', '/xim/da/hm10/t']

    # t_win = [0.28, 0.29]
    # t_win = [0.075, 0.085]
    t_win = [0.15, 0.155]
    # t_win = [0.345, 0.355]

    t_slice = dict(slice_=dict(t=slice(*t_win)))

    # pulse = 43596
    pulse = 43591
    # pulses = [43596, 43643, 43644, ]
    # pulses = [43665, 43666 ]
    # pulses = [43611, 43613, 43614]  # KPI fueling, flux expansion scan
    # pulses = [43591]  # Potential clip T2 in superx


    pulses = [
        # (43591, dict(slice_=dict(t=slice(0.195, 0.20)))),
        # (43596, dict(slice_=dict(t=slice(0.195, 0.20)))),
        # (43591, dict(slice_=dict(t=slice(0.335, 0.34)))),
        # (43596, dict(slice_=dict(t=slice(0.335, 0.34)))),

        # (43611, t_slice),
        # (43613, t_slice),
        # (43614, t_slice),

        # (43644, dict(slice_=dict(t=slice(0.20, 0.202)))), #R_path0=slice(*r_t2_bounds)))),
        # (43644, dict(slice_=dict(t=slice(0.24, 0.245)))), #R_path0=slice(*r_t2_bounds)))),
        # (43644, dict(slice_=dict(t=slice(0.325, 0.330)))), #R_path0=slice(*r_t5_bounds)))),

        # (43835, dict(slice_=dict(t=slice(0.606, 0.609)))), #R_path0=slice(*r_t5_bounds)))),
        # (43835, dict(slice_=dict(t=slice(0.69, 0.7)))), #R_path0=slice(*r_t5_bounds)))),
        # (43835, dict(slice_=dict(t=slice(0.744, 0.746)))), #R_path0=slice(*r_t5_bounds)))),
        # (43835, dict(slice_=dict(t=slice(0.82, 0.825)))), #R_path0=slice(*r_t5_bounds)))),

        # 43587,
        # (43587, dict(slice_=dict(t=0.070))),
        # (43587, dict(slice_=dict(t=0.075))),
        # (43587, dict(slice_=dict(t=0.080))),
        # (43587, dict(slice_=dict(t=slice(0.0750, 0.080)))),
        # (43587, dict(slice_=dict(t=0.100))),
        # (43587, dict(slice_=dict(t=0.125))),
        # (43587, dict(slice_=dict(t=0.280))),
              ]
    # pulses = [
    #     (43596, dict(slice_=dict(t=slice(0.295, 0.305)))),
    #     (43596, dict(slice_=dict(t=slice(0.335, 0.345)))),
    #     (43596, dict(slice_=dict(t=slice(0.365, 0.375)))),
    # ]

    # pulse = 43835
    # times = np.arange(0.4, 0.9, 0.05)  # 43835
    # times = [0.499, 0.550, 0.607, 0.648, 0.698, 0.700]  # 43835
    # pulse = 43852
    # times = np.arange(0.118, 0.200, 0.015)  # 43852
    # times = [0.165, 0.550, 0.607, 0.648, 0.698, 0.700]  # 43852
    # pulse = 43916
    # times = np.arange(0.1, 0.35, 0.025)  # 43916
    pulse = 43917
    times = np.arange(0.45, 0.65, 0.025)  # 43917
    dt_av = 0.0025
    # pulses = [(pulse, dict(slice_=dict(t=slice(t, t+dt_av)))) for t in times]
    pulses = [(p, {}) for p in np.arange(44021, 44026)]


    # signals_mixed = [['heat_flux_path1'],]
    signals_mixed = [['heat_flux_path0'],]
    # signals_mixed = [['temperature_path0'],]

    slice_ = dict(t=slice(*t_win))
    reduce = ('t', 'mean', ())
    # reduce = None
    x_coord = 'R_path0'
    # x_coord = 'R_path1'
    # x_range = (0.7, 0.9)  # for R on x axis, T2
    # x_range = (1.4, 1.75)  # for R on x axis, T5
    # x_range = (-0.05, 0.6)  # for t on x axis
    x_range = None
    debug_plots.plot_mixed_fire_uda_signals(signals_mixed, pulses=pulses, slice_=slice_, reduce=reduce, x_coord=x_coord,
                                            meta_data=dict(pulse=pulse, camera=camera, machine=machine),
                                            x_range=x_range, sharex='col', normalise_sigs_on_same_ax=False,
                                            separate_pulse_axes=False, recompute_pickle=False)
    plot_tools.show_if(True, tight_layout=True)

    if False:
        signals_mixed = [['heat_flux_peak_path0'],
                     ['heat_flux_r_peak_path0'],
                     # ['/ane/density'],
                     # [('xim/da/hm10/t', 'xim/da/hm10/r')],
                     # [['xpx/clock/lwir-1'], ['/AMC/ROGEXT/TF']],
                     ]
        debug_plots.plot_mixed_fire_uda_signals(signals_mixed, pulses=pulses,
                                                meta_data=dict(pulse=pulse, camera=camera, machine=machine),
                                                sharex='col', normalise_sigs_on_same_ax=False, separate_pulse_axes=False,
                                                recompute_pickle=False)
        # plot_tools.show_if(True, tight_layout=True)

    if False:
        signals_mixed = [[['heat_flux_peak_path0'], ['temperature_peak_path0']],
                         ['heat_flux_r_peak_path0', 'temperature_r_peak_path0'],
                         [('xim/da/hm10/t', 'xim/da/hm10/r')],
                         # [['xpx/clock/lwir-1'], ['/AMC/ROGEXT/TF']],
                         ]

        debug_plots.plot_mixed_fire_uda_signals(signals_mixed, pulses=pulses,
                                                meta_data=dict(pulse=pulse, camera=camera, machine=machine),
                                                sharex='col', normalise_sigs_on_same_ax=False)
        plot_tools.show_if(True, tight_layout=True)
        # raise

    if False:
        compare_shots_1d_with_uda_signals(camera, signals_ir=signals, signals_uda=signals_uda, pulses=pulses, machine=machine,
                                          show=False)


    if False:
        # signals = {'heat_flux_max(i)_path0': dict(x_coord='s_global_path0', offset_x_peak=True)}
        # offset_x_peak = True
        offset_x_peak = False
        signals = {'heat_flux_path0': dict(x_coord='R_path0', reduce=[('t', np.mean, ())], offset_x_peak=offset_x_peak)}

        # pulses = {43163: {}, 43183: {}}
        # pulses = {43163: dict(slice_=slice_), 43183: dict(slice_=slice_), 43415: dict(slice_=slice_)}

        # t_win = [0.28, 0.29]
        # t_win = [0.14, 0.15]
        t_win = [0.33, 0.34]

        r_win = [1.40, None]  # T5

        if t_win is not None:
            slice_ = {'t': slice(*t_win), 'R_path0': slice(*r_win)}
        else:
            slice_ = {'R_path0': slice(*r_win)}
        pulses_dict = {pulse: dict(slice_=slice_) for pulse in pulses}

        compare_shots_1d('rit', signals=signals, pulses=pulses_dict, show=False)

    signals = {'heat_flux': dict()}
    # signals = {'temperature': dict()}
    robust = True
    # robust = False
    t_win = None  # Don't label time window
    # t_range = [0, 0.35]
    t_range = [0, 0.7]
    # t_range = None

    # r_range=r_t5_bounds
    r_range = None

    colorbar_kwargs = {}
    # colorbar_kwargs = dict(vmin=-0.011, vmax=0.020, extend='neither')  # , extend='neither', 'both', 'max'
    compare_shots_2d(camera, signals=signals, pulses=pulses, machine=machine, machine_plugins=machine_plugins,
                     t_range=t_range, r_range=r_range, t_wins=t_win, robust=robust,
                     set_ax_lims_with_ranges=True, show=False, colorbar_kwargs=colorbar_kwargs,
                     robust_percentiles=(50, 99.5))

    plot_tools.show_if(True, tight_layout=True)

    pass