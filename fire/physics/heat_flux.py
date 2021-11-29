# -*- coding: future_fstrings -*-
#!/usr/bin/env python

"""


Created:
"""

import logging
from typing import Union, Iterable, Tuple, Optional, Sequence
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import matplotlib.pyplot as plt

from fire.interfaces.io_basic import read_csv
from fire.misc.utils import safe_len
from fire.misc import utils


logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

def theo_kwargs_for_path(material_ids, visible_materials, material_properties,
                  force_material_sub_index=None, raise_on_multiple_materials=True):
    if (force_material_sub_index is not None):
        material_id = force_material_sub_index
    else:
        if (len(visible_materials) > 1) and raise_on_multiple_materials:
            raise ValueError('Multiple materials for analysis path - beak up path')
        else:
            material_id = list(visible_materials.keys())[0]
    material_name = visible_materials[material_id]
    theo_kwargs = material_properties[material_name]
    return material_name, theo_kwargs

def scan_alpha_param(temperature_path, t, s_path, theo_kwargs, alpha_values, test=True, meta=(), verbose=True):
    from fire import theodor
    from fire.plotting import plot_tools
    print(f'Performing alpha scan with values: {alpha_values}')
    pulse, camera, machine = dict(meta).get('pulse'), dict(meta).get('camera'), dict(meta).get('machine')

    alpha_passed = theo_kwargs.get('alpha_top_org')

    fig, axes, ax_passed = plot_tools.get_fig_ax(num=f'alpha_scan_heatmaps, {machine} {diag_tag_raw} {pulse}',
                                                 ax_grid_dims=(2, int(np.ceil(len(alpha_values)/2))),
                                                 figsize=(12, 12))
    axes = axes.flatten()
    data = {}
    stats = defaultdict(list)
    radial_average = defaultdict(list)
    radial_average['temperature_av'].append(np.mean(temperature_path, axis=0))
    radial_average['temperature_min'].append(np.min(temperature_path, axis=0))
    radial_average['temperature_max'].append(np.max(temperature_path, axis=0))
    radial_average['temperature_98%'].append(np.percentile(temperature_path, 98, axis=0))
    radial_average['temperature_95%'].append(np.percentile(temperature_path, 95, axis=0))

    for i, alpha in enumerate(alpha_values):
        theo_kwargs['alpha_top_org'] = alpha
        heat_flux, extra_results = theodor.theo_mul_33(temperature_path, t, s_path, test=test, verbose=verbose,
                                                       **theo_kwargs)
        heat_flux *= 1e-6
        data[tuple(alpha)] = heat_flux.T
        stats['max'].append(np.max(heat_flux))
        stats['99%'].append(np.percentile(heat_flux, 99))
        stats['98%'].append(np.percentile(heat_flux, 98))
        stats['mean'].append(np.mean(heat_flux))
        stats['median'].append(np.percentile(heat_flux, 50))
        stats['2%'].append(np.percentile(heat_flux, 2))
        stats['1%'].append(np.percentile(heat_flux, 1))
        stats['min'].append(np.min(heat_flux))
        radial_average['heat_flux'].append(np.mean(heat_flux, axis=0))
        ax = axes[i]
        im = ax.pcolor(heat_flux, cmap='coolwarm')
        plot_tools.annotate_axis(ax, rf'$\alpha=${alpha}', loc='top left')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
    ax.axvline(x=alpha_passed, ls='--', color='k')
    plot_tools.annotate_providence(ax, meta_data=meta)
    plot_tools.save_fig(fire_paths['figures'] / f'alpha_scan/heatmaps/'
                                                f'alpha_scan_heatmaps-{machine}-{diag_tag_analysed}-{pulse}.png', mkdir_depth=2)
    plot_tools.show_if(True)

    fig, ax, ax_passed = plot_tools.get_fig_ax(num=f'alpha_scan_q_stats, {machine} {diag_tag_analysed} {pulse}',
                                               ax_grid_dims=(1, 1))
    alphas_flat = [a[0] for a in alpha_values]
    for stat, values in stats.items():
        ax.plot(alphas_flat, values, marker='x', color=None, label=stat)
    # ax.plot(alphas_flat, medians, marker='.', color=None, label='median')
    # ax.plot(alphas_flat, maxs, marker='o', color=None, label='max')
    ax.axhline(y=0, ls=':', color='k')
    plot_tools.annotate_providence(ax, meta_data=meta)
    plot_tools.legend(ax)
    ax.set_xlabel(r'$\alpha$ [W/(m$^2$•K)]')
    ax.set_ylabel(r'$q_\perp$ [MW]')
    plot_tools.save_fig(fire_paths['figures'] / f'alpha_scan/q_stats/alpha_scan_q_stats-{machine}-{diag_tag_analysed}-{pulse}.png',
                        mkdir_depth=2)
    plot_tools.show_if(show=True, tight_layout=True)

    fig, axes, ax_passed = plot_tools.get_fig_ax(num=f'alpha_scan_radial_av, {machine} {diag_tag_analysed} {pulse}',
                                               ax_grid_dims=(1, 2), sharex=True)
    ax = axes[0]
    for alpha, profile in zip(alphas_flat, radial_average['heat_flux']):
        ax.plot(t, profile, marker='.', color=None, label=r'$q_{av}$, $\alpha$='+fr'{alpha:0.3g}', alpha=0.7)
    ax.set_ylabel('$q_{\perp,av}$ [MW/m$^2$]')
    plot_tools.annotate_providence(ax, meta_data=meta)
    plot_tools.legend(ax=ax)

    ax = axes[1]
    ax.plot(t, radial_average['temperature_max'][0], marker='.', ls='-', color=None, label=r'$T_{max}$')
    ax.plot(t, radial_average['temperature_98%'][0], marker='.', ls='-', color=None, label=r'$T_{98%}$')
    ax.plot(t, radial_average['temperature_95%'][0], marker='.', ls='-', color=None, label=r'$T_{95%}$')
    ax.plot(t, radial_average['temperature_av'][0], marker='.', ls='-', color='k', label=r'$T_{av}$')
    ax.plot(t, radial_average['temperature_min'][0], marker='.', ls='-', color=None, label=r'$T_{min}$')

    plot_tools.legend(ax=ax)

    ax.set_xlabel('$t$ [s]')
    ax.set_ylabel('$T$ [$^\circ$C]')
    plot_tools.save_fig(fire_paths['figures'] / f'alpha_scan/radial_av/'
                                                f'alpha_scan_radial_av-{machine}-{diag_tag_analysed}-{pulse}.png',mkdir_depth=2)
    plot_tools.show_if(show=True, tight_layout=True)


def calc_heatflux(t, temperatures, path_data, path_name, theo_kwargs, force_material_sub_index=None, meta=()):
    """

    Args:
        t: 1d array of time values
        temperatures: 3d array of frame temperatures (t, y_pix, x_pix)
        path_data: dataarray describing analysis path. Includes analysis path pixel coordinates and material indexes
        path_name: key name of analysis path eg 'path0'
        material_properties: dict of material properties for each material index
        visible_materials:
        force_material_sub_index: Material index to use for whole analysis path. Eg use to still analyse sections of
                                  analysis path with unknown material index (-1)

    Returns: heat_flux_2d(s_path, t), extra_results

    """
    from fire import theodor
    meta = dict(meta)

    t = np.array(t)
    path = path_name

    # Check theodor time axis is uniform
    dt = np.diff(t)
    dt_mode = stats.mode(dt).mode
    mask_const_dt = np.isclose(dt, dt_mode, atol=9e-7, rtol=5e-3)
    if not np.all(mask_const_dt):
        logger.warning(f'Time axis of data supplied to THEODOR is not uniform. mode={dt_mode}, '
                       f'n_diff={np.sum(~mask_const_dt)}. dt other: {dt[~mask_const_dt]}')
    # TODO: Check analysis path for jumps/tile gaps etc

    # TODO generalise to other path names (i.e. other than 'path')
    material_ids = np.array(path_data[f'material_id_{path}'])
    material_ids = set(material_ids[~np.isnan(material_ids)])
    if len(material_ids) == 0:
        raise ValueError(f'No surface materials identified along analysis path: {path}')
    elif -1 in material_ids:
        if (force_material_sub_index is not None) and (len(material_ids) == 2):
            pass  # Treat whole path as being material given by force_material_sub_index
        else:
            raise ValueError('Analysis path contains unknown materials')
    elif len(material_ids) > 1:
        raise NotImplementedError(f'Multiple materials along analysis path')
    # TODO: Loop over sections of path with different material properties or tile gaps etc
    xpix_path, ypix_path = path_data[f'x_pix_{path}'], path_data[f'y_pix_{path}']
    temperature_path = np.array(temperatures[:, ypix_path, xpix_path])

    s_path = np.array(path_data[f's_global_{path}'])  # spatial coordinate along tile surface
    if np.any(np.isnan(s_path)):
        logger.warning('s_global coordinate contains nans. Replacing with R')
        s_path = np.array(path_data[f'R_{path}'])
    if np.any(np.isnan(s_path)):
        s_path = utils.interpolate_out_nans(s_path)

    # TODO: Understand when two element alpha_top_org values should be used
    if False:
        if safe_len(theo_kwargs['alpha_top_org']) == 2:
            theo_kwargs['alpha_top_org'] = theo_kwargs['alpha_top_org'][0]
    else:
        # theo_mul_33 expects argument to be an ndarray (checks alpha_top_org.ndim)
        theo_kwargs['alpha_top_org'] = np.array(theo_kwargs['alpha_top_org'])

    if temperature_path.shape[0] != len(s_path):
        if temperature_path.shape[1] == len(s_path):
            temperature_path = temperature_path.T
            logger.warning(f'Transposed THEODOR temperature input data to start with spatial dimension (t,s) -> (s,t)')
        else:
            raise ValueError(f'Spatial dimension of temperature path data ({temperature_path.shape}) does not match '
                             f's_path dimension ({len(s_path)}). '
                             f'Mismatch will cause theodor to use integer index s values!')

    # alpha_bot = alphas['tile_bottom']
    # alpha_top = alphas['tile_surface']
    #
    # d_target=0.040
    # diff = np.array([ 70.63, 48.25, 37.78])*1.0e-6
    # lam = np.array([174.9274, 133.1148, 110.4595])
    # #diff = np.array([240.87, 61.53, 34.86])*1e-6    # mm^2/s    Valerias JET data, 01/12/2008
    # #lam  = np.array([305.28,175.68,117.12])         # W/m/K     Valerias JET data
    # aniso=1.00
    # alpha_bot=200.0
    # acl=alphavector if alphavector is not None else np.array([1.])
    # alpha_top = alpha*acl if alpha is not None else 220000.0*acl


    # For hints as to meanings to theodor arguments see:
    # https://users.euro-fusion.org/openwiki/index.php/THEODOR#theo_mul_33.28.29
    # NOTE: If location[x] is not the same size as the x dimension of data[x,t], the code uses the array indices as location[x].
    if False:
        # tmp imports for theodor debugging
        import faulthandler, gc
        faulthandler.enable()
        gc.disable()

    """
    Required arguments:
    data            - float temperature array (location,time) [C]
    time            - temporal coordinates of data[1] [s]
    location        - spatial coordinates of data[0] [m]
    d_target        - target thickness [m]
    alpha_bot       - heat transmission coefficient at the bottom of the tile [W/(m2•K)] 
    alpha_top_org   - heat transmission coefficient at the top of the tile. Not sure when this should have two elements 
    diff            - heat diffusion coefficient at [0,500,1000 C] [m^2/s]
    lam             - heat conduction coefficient (at [0,500,1000 C]) [W/m/K]
    aniso           - anisotropy of vertical and tangential heat conduction. Ratio of 1.0 -> isotropic
    
    NOTES:
    - The remaining arguments are for visualisation/debugging and can be ignored
    - If location[x] is not the same size as the x dimension of data[x,t], the code uses the array indices as location[x].
    """
    logger.info(f'Calling THEODOR with kwargs: {theo_kwargs}')

    heat_flux, extra_results = theodor.theo_mul_33(temperature_path, t, s_path, test=True, verbose=True, **theo_kwargs)
    #               d_target, alpha_bot, alpha_top, diff, lam, aniso, x_Tb=x_Tb, y_Tb=y_Tb,

    scan_alphas = False
    # scan_alphas = True
    if scan_alphas:
        # alpha_values = [[70000], [10000]]
        alpha_values = [[a] for a in np.arange(10e2, 250e3, 25e3)]
        alpha_values = [np.array(a) for a in alpha_values]
        scan_alpha_param(temperature_path, t, s_path, theo_kwargs, alpha_values, test=True, meta=meta, verbose=True)

    # Convert W -> MW
    heat_flux *= 1e-6

    # Check theo output
    mask_nans = np.isnan(heat_flux)
    if np.any(mask_nans):
        n_nans = np.sum(mask_nans)
        logger.warning(f'Heat flux data contains {n_nans}/{heat_flux.size} {n_nans/heat_flux.size:0.1%} nans')
        if n_nans/heat_flux.size > 0.1:
            raise ValueError(f'Heat flux output contains more than 10% nans: {n_nans/heat_flux.size:0.1%}')

    return heat_flux, extra_results

if __name__ == '__main__':
    pass