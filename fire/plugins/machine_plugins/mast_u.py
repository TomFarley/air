# -*- coding: future_fstrings -*-
"""MAST specific plugin functions for FIRE.

Functions and top level variables in this file with appropriate names will be imported and used by FIRE to provide
functionality and information specific to this machine.

The most important role of the machine plugin is to provide a means of looking up the tile surface 's' coordinate
against which heat flux profiles etc are defined. Additional machine specific information and tools include
identifying sector numbers and other location based labels and plotting methods for visualising this information.

Author: Tom Farley (tom.farley@ukaea.uk)
Created: 01-2020
"""
import logging
from copy import copy

import numpy as np
import xarray as xr
from fire.camera_tools.camera_checks import logger
from fire.interfaces import uda_utils
from fire.plotting.plot_tools import format_poloidal_plane_ax
from fire.geometry.s_coordinate import interpolate_rz_coords, separate_rz_points_top_bottom, calc_s_coord_lookup_table, \
    get_nearest_s_coordinates, get_nearest_rz_coordinates, get_nearest_boundary_coordinates

from fire.geometry.geometry import cartesian_to_toroidal
from fire.misc.utils import make_iterable
from fire.misc import data_structures, utils
from fire.plotting import plot_tools

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# ===================== PLUGIN MODULE ATTRIBUTES =====================
# Which plugin attributes/functions are required or optional is set in your fire_config.json file under "plugins".

# Required: Name of plugin module (typically name of machine == name of file), needed to be located as a plugin module
machine_plugin_name = 'mast_u'

# Recommended
machine_name = 'MAST-U'  # Will be cast to lower case (and '-' -> '_') in FIRE
plugin_info = {'description': 'This plugin supplies functions for MAST-U specific operations/information'}  # exta info
location_labels_im = ['sector', 's_global']  # Parameters used to label coordinates across the whole image

# Optional/other
n_sectors = 12  # Used internally in funcs below
# Machine specific
None

# Boxes to pass to fire.s_coordinate.remove_false_rz_surfaces
false_rz_surface_boxes_default = [((1.512, 1.6), (-0.81789, 0.81789)),]
# Origin for s coordinates/divider between top and bottom of machine
s_start_coord_default = (0.260841, 0)

# Use same plugin funcs for machine sector and s_path coordinate as for MAST
from fire.plugins.machine_plugins.mast import (get_machine_sector, get_tile_louvre_index, get_tile_louvre_label,
                                               get_s_coord_path)
# See bottom of file for function aliases
# ====================================================================

module_default = object()

# Radii of edges of different structures for top down view plot etc
surface_radii = {'R_wall': 2.0,
                'R_HL04': 2.1333,
                'R_T1': 0.333,
                'R_T2': 0.540,
                'R_T3': 0.905,
                'R_T4': 1.081,
                'R_T5': 1.379,
                'R_T5_top': 1.746,
                'n_sectors': 12,
                }

def get_wall_rz_coords(no_cal=True, signal="/limiter/efit", shot=50000, ds=None, use_mast_client=False):
    """Return (R, Z) coordinates of points defining wall outline of MAST-U tile surfaces

    This is normally safe to call with default arguments.

    Args:
        no_cal: Whether to return idealised CAD coordinates without spatial calibration corrections
        signal: UDA signal for wall coords
        shot: Shot number to get wall definition for
        use_mast_client: Whether to use mast uda client

    Returns: Tuple of (R, Z) coordinate arrays

    """
    # import pyuda
    from fire.interfaces import uda_utils
    # TODO: Enable returning wall for MAST ie shot < 50000
    # client=pyuda.Client()
    uda_module, client = uda_utils.get_uda_client(use_mast_client=use_mast_client, try_alternative=True)
    wall_data = client.geometry(signal, shot, no_cal=no_cal)
    r = wall_data.data.R
    z = wall_data.data.Z
    if ds is not None:
        r, z = interpolate_rz_coords(r, z, ds=ds, false_surface_boxes=false_rz_surface_boxes_default)
    return r, z

def get_tile_edge_coords_mastu(no_cal=True, signal="/limiter/efit", shot=50000, subset=True):
    """Return (R,Z) coords of main tile boundaries

    Args:
        no_cal: Whether to use idealised CAD coordinates without spatial calibration corrections
        signal: UDA signal for wall coords
        shot: Shot number to get wall definition for

    Returns: Tuple of (R, Z) coordinate arrays

    """
    min_tile_size = 0.10
    r0, z0 = get_wall_rz_coords(signal=signal, no_cal=no_cal, shot=shot)
    diff = np.linalg.norm([np.diff(r0), np.diff(z0)], axis=0)
    mask = (diff > min_tile_size) | (np.roll(diff, -1) > min_tile_size)
    mask = np.concatenate([[np.True_], mask])
    r_tiles, z_tiles = r0[mask], z0[mask]
    if subset:
        if subset is True:
            # Just pick out boundaries for tiles T1-T5
            n = len(r_tiles)
            subset = np.concatenate([np.arange(1, 9), np.arange(n-34, n-26)])
        r_tiles, z_tiles =  r_tiles[subset], z_tiles[subset]
    return r_tiles, z_tiles

def get_s_coords_tables_mastu(ds=1e-4, no_cal=True, signal="/limiter/efit", shot=50000):
    """Return dict of dataframes containing (R, Z, s) coordinates for top and bottom regions of the machine

    Args:
        ds: Resolution to interpolate wall coordinate spacing to in meters
        no_cal: Whether to use idealised CAD coordinates without spatial calibration corrections
        signal: UDA signal for wall coords
        shot: Shot number to get wall definition for

    Returns: Dict of dataframes containing (R, Z, s) coordinates for top and bottom regions of the machine

    """
    r0, z0 = get_wall_rz_coords(no_cal=no_cal, signal=signal, shot=shot)
    r, z = interpolate_rz_coords(r0, z0, ds=ds)
    (r_bottom, z_bottom), (r_top, z_top) = separate_rz_points_top_bottom(r, z, prepend_start_coord=True,
                                                                         bottom_coord_start=s_start_coord_default,
                                                                         top_coord_start=s_start_coord_default)

    s_bottom = calc_s_coord_lookup_table(r_bottom, z_bottom)
    s_top = calc_s_coord_lookup_table(r_top, z_top)

    s = {'s_bottom': s_bottom, 's_top': s_top}
    # s_bottom = s_bottom.to_xarray().rename({'s': 's_bottom'})
    # s_top = s_top.to_xarray().rename({'s': 's_top'})
    # s = xr.merge([s_bottom, s_top])  # , combine_attrs='no_conflicts')
    return s

# def get_s_coord_table_for_point():

# TODO: Import from mastcodes/uda/python/mast/geom/geomTileSurfaceUtils.py ? - from mast.geom.geomTileSurfaceUtils

def get_nearest_s_coordinates_mastu(r, z, tol=4e-3, ds=1e-3, no_cal=True, signal="/limiter/efit", shot=50000,
                                    tol_t5_ripple=25e-3, t5_edge_r_range=(1.388, 1.395), path_fn_geometry=None):
    """Return closest tile surface 's' coordinates for supplied (R, Z) coordinates

    Args:
        r: Array of radial R coordinates
        z: Array of vertical Z coordinates
        tol: Tolerance distance for points from wall - return nans if further away than tolerance.
             5mm is appropriate for most surfaces, but should be increased to 25mm+ for T5 to account for ripple shaping
             A larger threshold will mean more erroneous points (eg tile edges) will be assigned s coords
        ds: Resolution to interpolate wall coordinate spacing to in meters
        no_cal: Whether to use idealised CAD coordinates without spatial calibration corrections
        signal: UDA signal for wall coords
        shot: Shot number to get wall definition for
        path_fn_geometry: Path to file defining coordinates of different in-vessel structures

    Returns: Dict of s coordinates for top/bottom, (Array of 1/-1s for top/bottom of machine, Dict keying 1/-1 to s
             keys)

    """
    r, z = make_iterable(r, ndarray=True), make_iterable(z, ndarray=True)
    s_lookup = get_s_coords_tables_mastu(ds=ds, no_cal=no_cal, signal=signal, shot=shot)
    z_mask = (z <= 0)
    s = np.full_like(r, np.nan, dtype=float)
    position = np.full_like(r, np.nan, dtype=float)
    table_key = {-1: 's_bottom', 1: 's_top'}
    for mask_half, key, pos in zip([z_mask, ~z_mask], ['s_bottom', 's_top'], [-1, 1]):
        lookup_table = s_lookup[key]
        if np.any(mask_half) and (not np.all(np.isnan(r[mask_half]))):
            r_masked, z_masked = r[mask_half], z[mask_half]
            r_wall, z_wall, s_wall = lookup_table['R'], lookup_table['Z'], lookup_table['s']
            s[mask_half] = get_nearest_s_coordinates(r_masked, z_masked, r_wall, z_wall, s_wall, tol=tol)

            # At the lower edge of T5 the ripple correction shaping of T5 requires an extra high tolerance that is not
            # desirable elsewhere where tiles are smooth
            # t5_edge_mask = (r_masked > t5_edge_r_range[0]) & (r_masked < t5_edge_r_range[1])

            from fire.geometry import geometry
            path_fn, surface_coords = geometry.read_structure_coords(path_fn=path_fn_geometry, machine='mast_u')
            structure_coords_df, structure_coords_info = geometry.structure_coords_dict_to_df(surface_coords)
            visible_structures_tuple = geometry.identify_visible_structures(r_masked, np.zeros_like(r_masked),
                                                                            z_masked, structure_coords_df)
            visible_structures = visible_structures_tuple[0]
            t5_edge_ind = np.nonzero(mask_half)[0][visible_structures == 5]
            t5_edge_mask = np.zeros_like(r, dtype=bool)
            t5_edge_mask[t5_edge_ind] = True  # mask_im = t5_edge_mask.reshape((320, 256))
            if np.any(t5_edge_mask):
                s[t5_edge_mask] = get_nearest_s_coordinates(r[t5_edge_mask], z[t5_edge_mask], r_wall, z_wall, s_wall,
                                                            tol=tol_t5_ripple)

            position[mask_half] = pos
    return s, (position, table_key)

# def create_poloidal_cross_section_figure(nrow=1, ncol=1, cross_sec_axes=((0, 0),)):
#     fig, axes = plt.subplots(nrow, ncol)
#     if nrow==1 and ncol==1:
#         format_poloidal_plane_ax(axes)
#     else:
#         for ax_coord in cross_sec_axes:
#             ax = axes[slice(*ax_coord)]
#             format_poloidal_plane_ax(ax)
#     plt.tight_layout()
#     return fig, axes

def get_s_coord_global(x_im, y_im, z_im, **kwargs):
    """Return MAST-U tile s coordinates for all pixels in image.

    This MAST-U 's' coordinate starts at 0m mid-way up the centre column and increases along tile surfaces to a maximum
    value of 6.088473 m (top) and 6.088573 m (bottom).
    This 's' coordinate is considered 'global' as it is predefined for all (R, Z) surfaces as apposed to a 'local' s
    starting at 0m along a specific path.

    Args:
        x_im        : x coordinates of each pixel in image
        y_im        : y coordinates of each pixel in image
        z_im        : z coordinates of each pixel in image
        **kwargs    : Arguments passed to get_nearest_s_coordinates_mastu

    Returns: MAST-U tile 's' coordinate for each pixel in the image

    """
    x, y, z = (np.array(d).flatten() for d in (x_im, y_im, z_im))
    r = np.linalg.norm([x, y], axis=0)
    # phi = np.arctan2(y_im, x_im)
    s, (position, table_key) = get_nearest_s_coordinates_mastu(r, z, **kwargs)
    s_im = s.reshape(x_im.shape)
    # TODO: Check if masking with nans is required in any situations
    return s_im

def format_coord(coord, **kwargs):
    """
    MAST-U coordinate formatter, includes sector number.

    Args:
        coord: Array of (x, y, z) coordinates describing a point in the machine

    Returns: String describing the position of the point within the machine

    """
    x, y, z = coord[0], coord[1], coord[2]
    r, phi = cartesian_to_toroidal(x, y)

    sector = get_machine_sector(phi)

    formatted_coord = 'X,Y,Z: ( {:.3f} m , {:.3f} m , {:.3f} m )'.format(x, y, z)
    formatted_coord = formatted_coord + u'\nR,Z,\u03d5: ( {:.3f} m , {:.3f}m , {:.1f}\xb0 )'.format(r, z, phi)
    formatted_coord = formatted_coord + '\n Sector {:.0f}'.format(sector)

    return formatted_coord


def plot_poloidal_tile_edges_mastu(ax=None, top=True, bottom=True, show=True, **kwargs):
    import matplotlib.pyplot as plt
    if ax is None:
        fig, ax = create_poloidal_cross_section_figure(1, 1)
    else:
        fig = ax.figure

    r0, z0 = get_tile_edge_coords_mastu(no_cal=True, shot=50000)

    mask = np.ones_like(r0, dtype=bool)
    mask = mask * (z0 > 0) if not bottom else mask
    mask = mask * (z0 < 0) if not top else mask

    ax.plot(r0[mask], z0[mask], marker='x', ls='', **kwargs)

    if show:
        plt.show()
    return fig, ax

def plot_vessel_outline(ax=None, top=True, bottom=True, shot=50000, no_cal=False, aspect='equal', ax_labels=True,
                              axes_off=False, show=True, **kwargs):
    """ Plot vessel outline for MAST-U

    Args:
        ax:
        top:
        bottom:
        shot:
        no_cal:
        aspect:
        ax_labels:
        axes_off:
        show:
        **kwargs:

    Returns:

    """
    from fire.plotting.plot_tools import plot_vessel_outline

    r0, z0 = get_wall_rz_coords(no_cal=no_cal, shot=shot)

    fig, ax = plot_vessel_outline(r0, z0, ax=ax, top=top, bottom=bottom, aspect=aspect, ax_labels=ax_labels,
                                axes_off=axes_off, show=show)

    return fig, ax

def plot_vessel_top_down(ax=None, keys_plot=('R_T1', 'R_T2', 'R_T3', 'R_T4', 'R_T5', 'R_T5_top', 'R_HL04'),
                         keys_plot_strong=('R_T1', 'R_T4', 'R_T5', 'R_T5_top'),
                         louvres=((('louvres_per_sector', 2), ('radii', ('R_T1', 'R_T5'))),
                                  (('louvres_per_sector', 4), ('radii', ('R_T5', 'R_T5_top')))),
                         axes_off=False, phi_labels=True):

    from fire.plotting.plot_tools import plot_vessel_top_down

    fig, ax = plot_vessel_top_down(surface_radii, keys_plot, ax=ax, axes_off=axes_off, phi_labels=phi_labels,
                                   keys_plot_strong=keys_plot_strong, louvres=louvres)
    return fig, ax

def label_tiles(ax, coords, coords_axes=('i_path{path_no}', 't'),
                tiles=('R_T1', 'R_T2', 'R_T3', 'R_T4', 'R_T5', 'R_T5_top'),
                coord_2_pos=0.01, plot_kwargs=None, path_no=0, legend_labels=False):
    # TODO: Make general machine agnostic wrapper function
    # TODO: Split into functions to identify visible tiles given r coordinate and labelling function
    if hasattr(ax, 'fire_meta'):
        if ax.fire_meta.get('has_tile_labels'):
            return  # These axes already have tile labels
    else:
        ax.fire_meta = {}

    plot_kws = dict(ls=':', lw=1.5, color='k', alpha=0.5)
    if plot_kwargs is not None:
        plot_kws.update(plot_kwargs)

    coords = copy(coords)
    coords_axes = make_iterable(coords_axes)
    coords_axes = tuple((value.format(path_no=path_no) for value in coords_axes))
    tile_labels = {'R_T1': 'T1', 'R_T2': 'T2', 'R_T3': 'T3', 'R_T4': 'T4', 'R_T5': 'T5', 'R_T5_top': ''}

    # if isinstance(coords, xr.core.coordinates.DataArrayCoordinates):

    # TODO: generalise path no.
    if (len(coords_axes) == 2):
        coord_2 = coords_axes[1]  # y axis variable, eg 'R_path0' for heat flux map plot
        coord_r = f'R_path{path_no}'
        coord_mono = f'i_path{path_no}'
        # Switch to monotonic coordinate compatible with sel 'nearest' eg i_path
        coord_2_data = data_structures.swap_xarray_dim(coords[coord_2],
                                                       new_active_dims=coord_mono, old_active_dims=coord_2)
        if coord_2_pos == 'min':
            coord_2_pos = coords[coords_axes[0]].min()
    else:
        coord_2 = coords.name  # variable is on y axis eg r_peak
        coord_r = coords_axes[0]
        coord_mono = coords.dims[0]
        # Switch to monotonic coordinate compatible with sel 'nearest' eg i_path
        coord_2_data = data_structures.swap_xarray_dim(coords[coord_r], new_active_dims=coord_mono, old_active_dims=coord_2)
        if coord_2_pos == 'min':
            coord_2_pos = coords.min()

    r = coords[coord_r].values
    r_range = [np.min(r), np.max(r)]
    r_tiles = np.array([surface_radii[key] for key in tiles])

    # Just keep tiles in R range
    tiles_visible = [(r_i > r_range[0] and r_i < r_range[1]) for r_i in r_tiles]
    tiles = np.array(tiles)[tiles_visible]
    r_tiles = r_tiles[tiles_visible]

    if (coord_2 == coord_mono):
        mask = np.argmin(np.abs(np.tile(r, (len(tiles), 1)).T - r_tiles), axis=0)
        tile_coord_values = coord_2_data.sel({coord_2: mask}, method='nearest')
    else:  #  (coord == coord_r):
        mask = np.argmin(np.abs(np.tile(r, (len(tiles), 1)).T - r_tiles), axis=0)
        tile_coord_values = coord_2_data.sel({coord_mono: mask}, method='nearest')
        # x = coords[coord].sel(**{coord: r_tiles}, method='nearest')



    for tile_coord_i, key in zip(tile_coord_values, tiles):
        label = tile_labels[key]
        if len(coords_axes) == 2 and (coord_r == coords_axes[1]):
            # label x axis
            ax.axvline(tile_coord_i, label=label if legend_labels else None, **plot_kws)
            plot_tools.annotate_axis(ax, label, loc=None, x=tile_coord_i, y=coord_2_pos, coords=('data', 'axes'),
                                     horizontalalignment='left', verticalalignment='bottom', box=False, clip_on=True)
        else:
            # label y axis
            ax.axhline(tile_coord_i, label=label if legend_labels else None, **plot_kws)
            plot_tools.annotate_axis(ax, label, loc=None, x=coord_2_pos, y=tile_coord_i, coords=('axes', 'data'),
                                     horizontalalignment='left', verticalalignment='bottom', box=False, clip_on=True)
    # else:
    #     raise NotImplementedError

    # Mark axes as being labeled so labelling isn't repeated
    ax.fire_meta['has_tile_labels'] = True

    return tiles, r_tiles

def pulse_meta_data(shot, keys=
                    # ('exp_date', 'exp_time', 'preshot', 'postshot', 'creation', 'abort', 'useful', 'sl',
                    #            'term_code', 'tstart', 'tend','program', 'reference')
                    #     ('tstart', 'tend', 'ip_max', 'tftstart', 'tftend', 'bt_max', 'bt_ipmax', 'pohm_max',
                    #      'tstart_nbi', 'tend_nbi', 'pnbi_max', 'rmag_efit', 'te0_max', 'ngreenwald_ipmax',
                    #      'ngreenwaldratio_ipmax', 'q95_ipmax', 'zeff_max', 'zeff_ipmax', 'betmhd_ipmax', 'zmag_efit',
                    #      'rinner_efit', 'router_efit')
('tstart', 'tend', 'ip_max', 'tftstart', 'tftend', )
                    ):
    try:
        from pycpf import pycpf
    except ImportError as e:
        return e

    out = dict()
    for key in keys:
        try:
            out[key] = pycpf.query([key], filters=['exp_number = {}'.format(shot)])[key]   # [0]
            out['erc'] = 0
        except Exception as e:
            logger.warning(f'Failed to perform CPF query for {shot}, {key}')
            if key in ('exp_date', 'exp_time'):
                try:
                    client = uda_utils.get_uda_client()
                    out['exp_date'], out['exp_time'] = client.get_shot_date_time(shot=shot)
                except Exception as e:
                    logger.warning('Also failed to retrieve date and time from UDA')
    return out

    # from matplotlib import patches
    # from fire.plotting.plot_tools import get_fig_ax
    # from fire.geometry.geometry import cylindrical_to_cartesian
    # fig, ax, ax_passed = get_fig_ax(ax)
    #
    # r_wall = surface_radii['R_wall']
    # n_sectors = surface_radii['n_sectors']
    #
    # # Plot vessel
    # wall = patches.Circle((0, 0), radius=r_wall, facecolor='b', edgecolor='k', alpha=0.1)
    # ax.add_patch(wall)
    #
    # # Plot tile radii etc
    # for key in ['R_T1', 'R_T2', 'R_T3', 'R_T4', 'R_T5', 'R_T5_top', 'R_HL04']:
    #     r = surface_radii[key]
    #     alpha = 0.6 if key in ['R_T1', 'R_T4', 'R_T5', 'R_T5_top'] else 0.3
    #     wall = patches.Circle((0, 0), radius=r, facecolor=None, fill=False, edgecolor='k', ls='--', alpha=alpha)
    #     ax.add_patch(wall)
    #
    # # Lines between sectors
    # for i in np.arange(n_sectors):
    #     x, y = cylindrical_to_cartesian(r_wall, i*360/n_sectors, angles_units='degrees')
    #     ax.plot([0, x], [0, y], ls='--', c='k', lw=1)
    #
    # # Sector numbers
    # for i in np.arange(n_sectors):
    #     x, y = cylindrical_to_cartesian(r_wall*0.9, 90-(i+0.5)*360/n_sectors, angles_units='degrees')
    #     ax.text(x, y, f'{i+1}', horizontalalignment='center', verticalalignment='center')
    #
    # # Phi labels
    # if phi_labels:
    #     for i in np.arange(4):
    #         phi = i*360/4
    #         x, y = cylindrical_to_cartesian(r_wall, phi, angles_units='degrees')
    #         label = f'$\phi={phi:0.0f}^\circ$'
    #         # annotate_axis(ax, label, x, y, fontsize=16, color='k')
    #         ax.text(x, y, label, horizontalalignment='left', verticalalignment='bottom')
    #     ax_scale = 1.2
    # else:
    #     ax_scale = 1.1
    #
    # ax.set_aspect(1)
    # ax.set_xlim(-r_wall * ax_scale, r_wall * ax_scale)
    # ax.set_ylim(-r_wall * ax_scale, r_wall * ax_scale)
    # if axes_off:
    #     ax.set_axis_off()
    # else:
    #     ax.set_xlabel(r'x [m]')
    #     ax.set_ylabel(r'y [m]')
    #     # plt.tight_layout()

def get_camera_external_clock_info(camera, pulse):
    if camera.lower() in ['rit', 'ait']:
        signal_clock = 'xpx/clock/lwir-1'
    elif camera.lower() in ['rir', 'air']:
        signal_clock = 'xpx/clock/mwir-1'
    else:
        raise ValueError(f'Camera name "{camera}" not supported')

    info_out = {}

    try:
        data = uda_utils.read_uda_signal_to_dataarray(signal_clock, pulse=pulse, raise_exceptions=True)
    except (Exception, RuntimeError) as e:
        logger.warning(e)
        return None
    else:
        if isinstance(data, Exception):
            return None
        data = data.astype(float)  # Switch from uint8

    t = data['t']

    signal_dv = np.concatenate([[0], np.diff(data)])

    frame_times = t[signal_dv > 1e-5]  # rising edges

    dt_frame = utils.mode_simple(np.diff(frame_times))  # time between frame aquisitions
    clock_freq = 1/dt_frame

    clock_peak_width = np.diff(t[np.abs(signal_dv) > 1e-5][:2])[0]  # Width of first square wave

    dt_signal = 1e-4
    data = data.interp(t=np.arange(t.min(), t.max(), dt_signal))

    # dt_signal = stats.mode(data['t'].diff(dim='t')).mode
    # t_high = data.sel(t=(data > data.mean()).values)['t']
    t_high = data['t'][(data > data.mean()).values]

    power = np.fft.fft(data)
    freq = np.fft.fftfreq(data.shape[-1], d=dt_signal)

    mask_pos = freq > 0
    power = power[mask_pos]
    freq = freq[mask_pos]

    power_abs = np.absolute(power)
    # plt.plot(freq, power.real, freq, power.imag)

    clock_freq_fft = freq[np.argmax(power_abs)]  # TODO: Convert freq to int?

    info_out['clock_frame_times'] = frame_times  # rising edges
    info_out['clock_t_window'] = np.array([frame_times[0], frame_times[-1]])
    info_out['clock_nframes'] = len(frame_times)
    info_out['clock_frequency'] = clock_freq
    info_out['clock_inter_frame'] = dt_frame
    info_out['clock_square_wave_width'] = clock_peak_width
    info_out['clock_frequency_fft'] = clock_freq_fft

    return info_out


def get_frame_time_correction(frame_times_camera, frame_times_clock=None, clock_info=None,
                              singals_da=('xim/da/hm10/t', 'xim/da/hm10/r')):
    """Work out corrected frame times/frame rate

    Problem can occur when manually recording camera data, that the camera internal frame rate is configured
    differently to the externally supplied clock rate. In this situation the frame times drift out of sequence with
    the clock. Presumably the frames are aquired on the first internal clock cycle occurring after a trigger signal?
    Alternatively could look into only internal clock times that occur during clock high values (+5V top of square wave)

    Args:
        frame_times_camera:
        frame_times_clock:
        clock_info:
        singals_da:

    Returns:

    """
    time_correction = {}
    if frame_times_clock is not None:
        frame_times_corrected = np.full_like(frame_times_clock, np.nan)
        for i, t_clock in enumerate(np.array(frame_times_clock)):
            frame_times_corrected[i] = frame_times_camera[frame_times_camera >= t_clock][0]
        dt_mean = utils.mode_simple(np.diff(frame_times_corrected))
        fps_dt_mean = 1/dt_mean
        fps_clock = 1/utils.mode_simple(np.diff(frame_times_clock))
        fps_camera = 1/utils.mode_simple(np.diff(frame_times_camera))
        factor = fps_dt_mean / fps_camera
    time_correction = dict(factor=factor, frame_times_corrected=frame_times_corrected, fps_dt_mean=fps_dt_mean,
                           fps_clock=fps_clock, fps_camera=fps_camera)
    return time_correction


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from fire.plotting.plot_tools import create_poloidal_cross_section_figure
    fig, ax = create_poloidal_cross_section_figure()
    plot_poloidal_tile_edges_mastu(ax=ax, show=True)
    ds = 1e-4
    # ds = 1e-2
    r0, z0 = get_wall_rz_coords(no_cal=True, shot=50000)
    r, z = interpolate_rz_coords(r0, z0, ds=ds, false_surface_boxes=false_rz_surface_boxes_default)
    print(f'Number of interpolated wall points: {len(r)}')
    s_tables = get_s_coords_tables_mastu()
    # point = (1.1, 0.9)
    # point = (1, 1)
    point = (1.1, -1.8)
    s_close, pos, table_key = get_nearest_s_coordinates_mastu(*point)
    s_close_table = s_tables[table_key[pos[0]]]
    wall_s_close = get_nearest_rz_coordinates(s_close, s_close_table['R'].values, s_close_table['Z'].values,
                                              s_close_table['s'].values)
    wall_close = get_nearest_boundary_coordinates(*point, r_boundary=r, z_boundary=z)[0]
    fig, ax = plt.subplots(1, 1)
    ax.plot(r, z, marker='x', ls='')
    ax.plot(r0, z0, marker='x', ls='')
    # ax.plot(r0[:10], z0[:10], marker='x', ls='')
    ax.plot(point[0], point[1], marker='x', ls='', c='r', label='point')
    ax.plot(wall_close[0], wall_close[1], marker='d', ls='', c='r', label='wall_close')
    ax.plot(wall_s_close[0], wall_s_close[1], marker='o', ls='', c='r', label='wall_s_close')
    ax.set_aspect('equal')
    ax.legend()
    plt.tight_layout()
    plt.show()
    pass