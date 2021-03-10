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
                     machine_plugins=None, t_wins=None, show=True):

    data = read_data_for_pulses_pickle(camera, pulses, machine)


    fig, axes, ax_passed = plot_tools.get_fig_ax(ax_grid_dims=(1, len(make_iterable(pulses))), axes_flatten=True,
                                                 figsize=(14, 8), sharex=True, sharey=True)
    path_key = 'path0'
    for signal, signal_props in signals.items():
        for ax, pulse in zip(axes, pulses):
            data_paths = data[pulse][0]['path_data']
            meta = data[pulse][0]['meta_data']

            # meta = dict(pulse=pulse, camera=camera, machine=machine)

            meta['path_label'] = meta['analysis_path_labels'][meta['analysis_path_keys'].index(path_key)]
            debug_plots.debug_plot_profile_2d(data_paths, param=signal, ax=ax, path_names=path_key, robust=True,
                                              annotate=True, meta=meta, t_wins=t_wins,
                                              machine_plugins=machine_plugins, show=False,
                                              colorbar_kwargs=dict(position='top', size='3%'))

    plt.tight_layout()
    plot_tools.show_if(show, tight_layout=False)
    plot_tools.save_fig(None)

if __name__ == '__main__':
    machine = 'mast_u'
    camera = 'rit'

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

    pulse = 43596
    pulses = [43596, 43643, 43644, ]

    signals_mixed = [['heat_flux_peak_path0'],
                     ['heat_flux_r_peak_path0'],
                     [('xim/da/hm10/t', 'xim/da/hm10/r')],
                     # [['xpx/clock/lwir-1'], ['/AMC/ROGEXT/TF']],
                     ]
    debug_plots.plot_mixed_fire_uda_signals(signals_mixed, pulses=pulses,
                                            meta_data=dict(pulse=pulse, camera=camera, machine=machine),
                                            sharex='col', normalise_sigs_on_same_ax=False, separate_pulse_axes=False)
    # plot_tools.show_if(True, tight_layout=True)

    signals_mixed = [[['heat_flux_peak_path0'], ['temperature_peak_path0']],
                     ['heat_flux_r_peak_path0', 'temperature_r_peak_path0'],
                     [('xim/da/hm10/t', 'xim/da/hm10/r')],
                     # [['xpx/clock/lwir-1'], ['/AMC/ROGEXT/TF']],
                     ]

    debug_plots.plot_mixed_fire_uda_signals(signals_mixed, pulses=pulses,
                                            meta_data=dict(pulse=pulse, camera=camera, machine=machine),
                                            sharex='col', normalise_sigs_on_same_ax=False)
    # plot_tools.show_if(True, tight_layout=True)
    # raise

    if True:
        compare_shots_1d_with_uda_signals(camera, signals_ir=signals, signals_uda=signals_uda, pulses=pulses, machine=machine,
                                          show=False)





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
    slice_ = {'t': slice(*t_win), 'R_path0': slice(*r_win)}
    pulses_dict = {pulse: dict(slice_=slice_) for pulse in pulses}

    compare_shots_1d('rit', signals=signals, pulses=pulses_dict, show=False)

    signals = {'heat_flux': dict()}
    compare_shots_2d(camera, signals=signals, pulses=pulses, machine=machine, machine_plugins=machine_plugins,
                     t_wins=t_win, show=False)

    plot_tools.show_if(True, tight_layout=True)

    pass