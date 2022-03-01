#!/usr/bin/env python

"""


Created: 
"""

import logging, os
from typing import Union, Iterable, Sequence, Tuple, Optional, Any, Dict
from pathlib import Path
from copy import copy

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

from fire.interfaces import io_utils
from fire.misc import utils
from fire.misc.utils import mkdir, make_iterable, is_scalar

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


def get_fig_ax(ax=None, num=None, ax_grid_dims=(1, 1), dimensions=2, axes_flatten=False,
               figsize=None, sharex='none', sharey='none', **kwargs):
    """If passed None for the ax keyword, return new figure and axes

    Args:
        ax: Existing axis instance to return figure instance for. If None, a new figure and axis is returned.
        num: Name of new figure window (if ax=None)
        ax_grid_dims: Dimensions of new figure axis grid (nrows, ncols) (if ax=None)
        dimensions: Number of dimensions axes for axis (2/3)
        axes_flatten: Return axes instances in a flattened 1d ndarray (irrespective of grid dims)
        **kwargs: Additional kwargs passed to plt.subplots

    Returns: fig (figure instance), ax (axis instance), ax_passed (bool specifying if ax was passed or generated here)

    """
    ax_grid_dims = np.array(ax_grid_dims)
    if ax is None:
        if dimensions == 3:
            if np.sum(ax_grid_dims) > 2:
                raise NotImplementedError
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
            fig = plt.figure(num=num)
            axes = fig.add_subplot(1, 1, 1, projection='3d')
        else:
            n_rows, n_cols = ax_grid_dims

            # subplots_adjust doesn't work with constrained_layout?
            constrained_layout = (sharex is 'none') and (sharey is 'none')

            fig, axes = plt.subplots(*ax_grid_dims, num=num, constrained_layout=constrained_layout,
                                   sharex=sharex, sharey=sharey, **kwargs)

            if np.any(ax_grid_dims > 1):
                for i_row in np.arange(n_rows):
                    for i_col in np.arange(n_cols):
                        if n_rows == 1:
                            ax = axes[i_col]
                        elif n_cols == 1:
                            ax = axes[i_row]
                        else:
                            ax = axes[i_row, i_col]
                        if (sharex in ('col', True)) and (i_row != n_rows-1):
                            ax.xaxis.label.set_visible(False)  # x axis labels only on bottom row
                            fig.subplots_adjust(hspace=0.03)  # reduce axes vertical height spacing
                            fig.tight_layout_off = True
                        elif (sharey in ('row', True)) and (i_col != 0):
                            ax.yaxis.label.set_visible(False)  # y axis labels only on left col
                            fig.subplots_adjust(wspace=0.03)  # reduce axes horizontal width spacing
                            fig.tight_layout_off = True



        ax_passed = False
    else:
        fig = ax.figure
        axes = ax
        ax_passed = True

    if axes_flatten:  # and isinstance(ax, (np.ndarray)):
        axes = make_iterable(axes, ndarray=True)
        axes = axes.flatten()

    return fig, axes, ax_passed

def get_ax_grid_dims(n_ax, n_max_ax_per_row=3):

    n_rows = int(np.ceil(n_ax / n_max_ax_per_row))
    n_cols = n_ax if n_rows == 1 else int(np.ceil(n_ax/n_rows))
    ax_grid_dims = (n_rows, n_cols)
    return ax_grid_dims

def add_second_y_scale(ax_left, color_left='tab:blue', color_right='tab:orange', label_right=None, dummy_values=None,
                       apply=True, ax_kwargs=None):
    """Add a second y axis at right of plot axes

    Args:
        ax: mpl axis instance to add y axis to
        y_values: New y axis values corresponding to original x axis values
        y_map_func:
        ax_kwargs:

    Returns:

    """
    # colors = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red')

    if not apply:
        return ax_left, None

    ax_right = ax_left.twinx()

    if dummy_values is not None:
        line, = ax_right.plot(dummy_values[0], dummy_values[1], label=None)  # Create a dummy plot
        line.set_visible(False)
        # ax2.cla()

    ax_right.set_xlabel(label_right)

    ax_left.tick_params(axis='y', labelcolor=color_left)
    ax_right.tick_params(axis='y', labelcolor=color_right)

    if ax_kwargs is not None:
        # Format axis ticks etc
        raise NotImplementedError

    return ax_right, ax_left

def add_second_x_scale(ax, x_axis_values, y_values=None, label=None, x_map_func=None, ax_kwargs=None):
    """Add a second x axis at top of plot axes

    Args:
        ax: mpl axis instance to add x axis to
        x_axis_values: New x axis values corresponding to original x axis values
        y_values: Dummy y values
        x_map_func:
        ax_kwargs:

    Returns:

    """
    x_axis_values = np.array(x_axis_values)
    if x_map_func is not None:
        x_axis_values = x_map_func(x_axis_values)
    if y_values is None:
        y_values = np.full_like(x_axis_values, np.nan, dtype=float)

    ax2 = ax.twiny()
    line, = ax2.plot(x_axis_values, y_values, label=None)  # Create a dummy plot
    line.set_visible(False)
    # ax2.cla()

    ax2.set_xlabel(label)

    if ax_kwargs is not None:
        # Format axis ticks etc
        raise NotImplementedError
    return ax2

def annotate_axis(ax, string, loc='top_right', x=0.85, y=0.955, fontsize=14, coords='axis', box=True,
                  bbox=(('facecolor', 'w'), ('ec', None), ('lw', 0), ('alpha', 0.5), ('boxstyle', 'round')),
                  horizontalalignment='center', verticalalignment='center', multialignment='center',
                  sub_underscores=True, margin=0.035, clip_on=True, **kwargs):
    """

    Args:
        ax:
        string:
        loc:
        x:
        y:
        fontsize:
        coords: Coordinate system of supplied location
        box:
        bbox:
        horizontalalignment:
        verticalalignment:
        multialignment:
        sub_underscores:
        margin:
        **kwargs:

    Returns:

    """
    if isinstance(bbox, (tuple, list)):
        bbox = dict(bbox)
    elif isinstance(bbox, dict):
        bbox_user = bbox
        bbox = dict((('facecolor', 'w'), ('ec', None), ('lw', 0), ('alpha', 0.5), ('boxstyle', 'round')))
        bbox.update(bbox_user)

    if sub_underscores:
        string = string.replace('_', ' ')

    if loc is None:
        pass  # Use passed x,y and alignment values
    else:
        loc = loc.replace(' ', '_').lower()
        loc_y, loc_x = loc.split('_')

        if loc_y == 'top':
            y = 1-margin
            verticalalignment = 'top'
        elif loc_y == 'bottom':
            y = margin
            verticalalignment = 'bottom'
        elif loc_y == 'middle':
            y = 0.5
            verticalalignment = 'centre'
        else:
            raise NotImplementedError(f'Annotate y location: "{loc_y}"')

        if loc_x == 'right':
            x = 1-margin
            horizontalalignment = 'right'
            multialignment = 'right'
        elif loc_x == 'left':
            x = margin
            horizontalalignment = 'left'
            multialignment = 'left'
        elif loc_x in ('centre', 'center'):
            x = 0.5
            horizontalalignment = 'center'
            multialignment = 'center'
        else:
            raise NotImplementedError(f'Annotate x location: "{loc_x}"')

    trans_dict = dict(axis=ax.transAxes, axes=ax.transAxes, data=ax.transData)
    if isinstance(coords, str):
        transform = trans_dict[coords]
    elif isinstance(coords, (tuple, list)):
        transform = [trans_dict[coord] if isinstance(coord, str) else coord for coord in coords]
        transform = transforms.blended_transform_factory(*transform)

    if not box:
        bbox = None

    artist = ax.text(x, y, string, fontsize=fontsize, bbox=bbox, transform=transform, clip_on=clip_on,
            horizontalalignment=horizontalalignment, verticalalignment=verticalalignment, multialignment=multialignment,
            **kwargs)
    return artist

def annotate_providence(ax, label='{machine} {pulse} {diag_tag_analysed}', loc='top_right',
                        meta_data=None, aliases=None, fontsize=12,
                        upper_case=('machine', 'diag_tag_analysed', 'diag_tag_raw'),
                        replace_chars=(('_', '-'), ),cut_strings='_path0', annotate=True, **kwargs):
    if not annotate:
        return

    backup_labels = [
        '{machine} {pulse} {diag_tag_analysed}\n({path_label})'
        '{machine} {pulse} {diag_tag_analysed} {path_label}',
        '{machine} {pulse} {diag_tag_analysed}',
        '{machine} {pulse} {diag_tag_raw}',
        '{pulse} {diag_tag_analysed}',
        '{pulse} {diag_tag_raw}',
        '{pulse}',
    ]

    if isinstance(ax, (np.ndarray, list, tuple)):
        ax = np.array(ax).flatten()[0]  # Only annotate first (top left) axis

    # TODO: split into meta prep function
    if meta_data is not None:
        if isinstance(meta_data, (xr.DataArray, xr.Dataset)):
            meta_dict = copy(meta_data.attrs.get('meta', meta_data.attrs))
            meta_dict['param'] = meta_data.attrs.get('symbol', meta_data.name)
        else:
            meta_dict = copy(meta_data)

        if aliases is not None:
            for key, alias in aliases.items():
                if key in meta_dict:
                    meta_dict[alias] = meta_dict.get(key)

        if upper_case is not None:
            for key in make_iterable(upper_case):
                meta_dict[key] = meta_dict.get(key, '').upper()

        try:
            label = label.format(**meta_dict)
        except KeyError as e:
            for label_bkup in backup_labels:
                try:
                    label = label_bkup.format(**meta_dict)
                except Exception as e:
                    pass
                else:
                    break

        # label = data_plot.attrs.get('symbol', data_plot.name).replace('_', ' ').replace(f'_{path_name}', '')
        if cut_strings is not None:
            for string in make_iterable(cut_strings):
                label = label.replace(string, '')

        for replace_char in make_iterable(replace_chars):
            label = label.replace(*replace_char)
    else:
        meta_dict = None

    if '{' in label:
        logger.warning(f'Un-formatted annotation string: {label}, {meta_dict}')

    artist = annotate_axis(ax=ax, string=label, loc=loc, fontsize=fontsize, **kwargs)

    return label, artist

def legend(ax, handles=None, labels=None, legend=True, only_multiple_artists=True, box=True, zorder=None,
           n_handles_reformat=8, max_col_chars=15, **kwargs):
    """Finalise legends of each axes"""
    if legend:
        kws = {'loc': 'best'}  #'fontsize': 14,
        if box:
            kws['facecolor'] = 'white'
            kws['framealpha'] = 0.65
            kws['fancybox'] = True
        else:
            kws['fancybox'] = False
            kws['framealpha'] = 0
            kws['facecolor'] = 'none'
        leg = None
        try:
            handles_current, labels_current = ax.get_legend_handles_labels()
            modify_elements = (handles is not None) or (labels is not None)

            handles = handles_current if (handles is None) else handles
            labels = labels_current if (labels is None) else labels

            n_handles = len(handles)

            if n_handles >= n_handles_reformat:
                if len(labels[0]) <= max_col_chars:
                    kws['ncol'] = int(np.ceil(n_handles / n_handles_reformat))
                    kws['fontsize'] = kws.get('fontsize', 12) * 0.8
                else:
                    kws['fontsize'] = kws.get('fontsize', 12) * 0.6

            # Only produce legend if more than one  artist has a label
            if ((not only_multiple_artists) or (len(handles_current) > 1) or modify_elements):
                args = () if (not modify_elements) else (handles, labels)
                kws.update(kwargs)
                leg = ax.legend(*args, **kws)
                leg.set_draggable(True)
                if zorder is not None:
                    leg.set_zorder(zorder)
        except ValueError as e:
            #  https: // github.com / matplotlib / matplotlib / issues / 10053
            logger.error('Not sure how to avoid this error: {}'.format(e))
    else:
        leg = ax.legend()
        leg.remove()
    return leg

def format_label(string, substitutions=[('_', ' ')]):
    for old, new in substitutions:
        string = string.replace(old, new)
    return string

def close_all_mpl_plots(close_all=True, verbose=True):
    """Close all existing figure windows"""
    if close_all:
        nums = plt.get_fignums()
        for n in nums[:-1]:
            plt.close(n)
        if verbose:
            logger.info('Closed all mpl plot windows')

def show_if(show=True, close_all=False, tight_layout=True, save_fig_fn=None, save_image_fn=None, save_kwargs=None):
    """If show is true show plot. If clear all is true clear all plot windows before showing."""
    close_all_mpl_plots(close_all)

    if save_kwargs is None:
        save_kwargs = {}

    if save_fig_fn is not None:
        save_fig(save_fig_fn, **save_kwargs)

    if save_image_fn is not None:
        save_fig(save_image_fn, **save_kwargs)

    if show:
        if tight_layout:
            fig = plt.gcf()
            if not hasattr(fig, 'tight_layout_off') or fig.tight_layout_off is False:
                # Some functions will set fig.tight_layout_off = True if layout has been manually customised
                try:
                    fig.tight_layout()
                except AttributeError as e:  # mpl version error?
                    logger.exception('fig.tight_layout() failed')
        plt.show()

def save_fig(path_fn, fig=None, paths=None, transparent=True, bbox_inches='tight', dpi=90, save=True,
             image_formats=None, image_format_subdirs='subsequent',
             mkdir_depth=None, mkdir_start=None, description='', meta_dict=(), verbose=True):
    if (not save) or (path_fn is None):
        return False
    image_formats = make_iterable(image_formats, ignore_types=(None,))
    meta_dict = dict(meta_dict)

    if fig is None:
        fig = plt.gcf()

    if paths is not None:
        paths = [Path(str(path).format(**meta_dict)) for path in make_iterable(paths)]
    else:
        paths = [Path(Path(path_fn).expanduser().resolve().parent)]

    fn = Path(str(path_fn).format(**meta_dict)).name

    for path in paths:
        path_fn = (path / fn).expanduser().resolve()

        if (mkdir_depth is not None) or (mkdir_start is not None):
            mkdir(path, depth=mkdir_depth, start_dir=mkdir_start)

        if image_formats is None:
            ext = path_fn.suffix
            path_fns = {ext: path_fn}
        else:
            # Handle file names without extension with periods in
            path_fn0, ext = os.path.splitext(str(path_fn))
            path_fn0 = path_fn0 if len(ext) <= 4 else path_fn
            path_fns = {}
            for i, ext in enumerate(image_formats):
                ext_dot = '.'+ext if ext[:1] != '.' else ext
                path_fn = path_fn.with_suffix(ext_dot)
                if ((image_format_subdirs == 'all') or (image_format_subdirs is True) or
                        ((image_format_subdirs == 'subsequent') and (i > 0))):
                    path_fn = io_utils.insert_subdir_in_path(path_fn, ext, -1, create_dir=True)
                path_fns[ext] = path_fn
        for ext, path_fn in path_fns.items():
            try:
                fig.savefig(path_fn, bbox_inches=bbox_inches, transparent=transparent, dpi=dpi)
            except (RuntimeError, AttributeError, ValueError) as e:
                logger.exception('Failed to save plot to: {}, {}'.format(path_fn, e))
                # raise e
            except Exception as e:
                raise e
        if verbose:
            path_fns_str = path_fns if len(path_fns) > 1 else list(path_fns.values())[0]
            logger.info('Saved {} plot to: {}'.format(description, path_fns_str))
            print('Saved {} plot to:'.format(description))
            print(path_fns)

def save_image(path_fn, image_data, path=None, save=True, image_formats=None, image_format_subdirs='subsequent',
             mkdir_depth=None, mkdir_start=None, description='', verbose=True):
    from skimage import io, exposure, img_as_uint, img_as_float

    if (not save) or (path_fn is None):
        return False
    if path is not None:
        path_fn = os.path.join(path, path_fn)
    path_fn = os.path.realpath(os.path.expanduser(path_fn))
    # if not pos_path(path_fn, allow_relative=True):  # string path
    #     raise IOError('Not valid save path: {}'.format(path_fn))

    if (mkdir_depth is not None) or (mkdir_start is not None):
        mkdir(os.path.dirname(path_fn), depth=mkdir_depth, start_dir=mkdir_start)

    if image_formats is None:
        _, ext = os.path.splitext(path_fn)
        path_fns = {ext: path_fn}
    else:
        # Handle filesnames without extension with periods in
        path_fn0, ext = os.path.splitext(path_fn)
        path_fn0 = path_fn0 if len(ext) <= 4 else path_fn
        path_fns = {}
        for i, ext in enumerate(image_formats):
            path_fn = '{}.{}'.format(path_fn0, ext)
            if ((image_format_subdirs == 'all') or (image_format_subdirs is True) or
                    ((image_format_subdirs == 'subsequent') and (i > 0))):
                path_fn = io_utils.insert_subdir_in_path(path_fn, ext, -1, create_dir=True)
            path_fns[ext] = path_fn
    for ext, path_fn in path_fns.items():
        image = np.array(image_data)
        try:
            io.use_plugin('freeimage')  # Enable writing out to higher bit depth (not just 8 bit, 0-255)
            image = exposure.rescale_intensity(image, out_range='float')
            image = img_as_uint(image)
        except ValueError as e:
            logger.debug('Failed use "freeimage" plugin to write high bit-depth image. Scaling to 8-bit: {}'.format(
                path_fn))
            image *= 255/np.max(image)
            image = image.astype(int)
        try:
            io.imsave(path_fn, image)  # Data must be integer or float [-1, 1]
        except RuntimeError as e:
            logger.exception('Failed to save image to: {}'.format(path_fn))
            raise e
    if verbose:
        logger.info('Saved {} image to:\n{}'.format(description, path_fns))
        print('Saved {} image to: {}'.format(description, path_fns))

def get_previous_artist_color(ax=None, artist_ranking=('line', 'pathcollection'), artist_ranking_str=None):
    artist_type_options = ('line', 'pathcollection')
    if ax is None:
        ax = plt.gca()
    if (artist_ranking_str is not None) and any([a in artist_ranking_str for a in artist_type_options]):
        artist_ranking = [a for a in artist_type_options if a in artist_ranking_str]

    for artist_type in artist_ranking:
        assert artist_type in artist_type_options
        if artist_type == 'line' and len(ax.lines) != 0:
            artist = ax.lines[-1]
            color = artist.get_color()
            break
        elif artist_type == 'pathcollection' and len(ax.collections) != 0:
            artists = [a for a in ax.collections if isinstance(a, mpl.collections.PathCollection)]
            if len(artists) > 0:
                artist = artists[-1]
                color = artist.get_facecolor()[0]  # [:2]
                break
    else:
        logger.warning("Can't repeat line color - no previous lines or path collections on axis")
        return 'k'
    return color

def get_previous_line_color(ax=None):
    """Return color of previous line plotted to axis"""
    if ax is None:
        ax = plt.gca()
    if len(ax.lines) != 0:
        color = ax.lines[-1].get_color()
    else:
        logger.warning("Can't repeat line color - no previous lines or path collections on axis")
        return 'k'

    return color

def color_shade(color, percentage):
    """Crude implementation to make color darker or lighter until matplotlib function is available:
    https://github.com/matplotlib/matplotlib/pull/8895"""
    from matplotlib import colors as mcolors
    c = mcolors.ColorConverter().to_rgb(color)
    c = np.clip(np.array(c)+percentage/100., 0, 1)
    return c

def repeat_color(ax=None, shade_percentage=None, artist_string=None):
    if ax is None:
        ax = plt.gca()
    color = get_previous_artist_color(ax, artist_ranking_str=artist_string)
    if (shade_percentage is None) and (artist_string is not None):
        if '+' in artist_string or '-' in artist_string:
            shade_percentage = float(artist_string.split('+')[-1].split('-')[-1])
    if shade_percentage is not None:
        c = color_shade(color, shade_percentage)
    return c

def setup_xarray_colorbar_ax(ax, data_plot=None, robust=True, cbar_label=None, cmap=None, position='right',
                             add_colorbar=True, size="5%", pad=0.05, log_cmap=False, **kwargs):
    from matplotlib import ticker
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    kws = dict(robust=robust)
    if add_colorbar:
        # Force xarray generated colorbar to only be hieght of image axes and thinner
        divider = make_axes_locatable(ax)
        ax_cbar = divider.append_axes(position=position, size=size, pad=pad)
        kws.update(dict(cbar_kwargs=dict(cax=ax_cbar)))  # , extend='both'

        if position in ('top', 'bottom'):
            tick_locator = ticker.MaxNLocator(nbins=5)
            kws['cbar_kwargs']['orientation'] = 'horizontal'
            kws['cbar_kwargs']['ticklocation'] = position
            kws['cbar_kwargs']['ticks'] = tick_locator
            ax_cbar.xaxis.set_ticks_position(position)
            ax_cbar.xaxis.set_label_position(position)

        if cbar_label is not None:
            kws['cbar_kwargs']['label'] = cbar_label  # If not passed xarray auto generates from dataarray name/attrs

    kws.update(kwargs)

    if log_cmap:
        kws['norm'] = matplotlib.colors.LogNorm()

    if (data_plot is not None) and np.issubdtype(data_plot.dtype, np.int64) and (np.max(data_plot.values) < 20):
        # Integer/quantitative data
        bad_value = -1
        data_plot = xr.where(data_plot == bad_value, np.nan, data_plot)

        values = data_plot.values.flatten()
        values = set(list(values[~np.isnan(values)]))
        ticks = np.sort(list(values))
        kws['cbar_kwargs']['ticks'] = ticks
        kws['vmin'] = np.min(ticks)
        kws['vmax'] = np.max(ticks)
        if cmap is None:
            cmap = 'jet'
        cmap = mpl.cm.get_cmap(cmap, (ticks[-1]-ticks[0])+1)

    kws['cmap'] = cmap
    kws['add_colorbar'] = add_colorbar

    return kws

def label_axis_windows(windows, axis='y', labels=None, ax=None, line_kwargs=None):
    windows = make_iterable(windows)

    line_kws = dict(ls='--', lw=1, color='k')
    if isinstance(line_kwargs, dict):
        line_kws.update(line_kwargs)

    if labels is None:
        labels = [None for win in windows]

    if ax is None:
        ax = plt.gca()

    if axis in ('x', 0):
        func = ax.axvline
    elif axis in ('y', 1):
        func = ax.axhline
    else:
        raise ValueError

    for i_win, (window, label) in enumerate(zip(windows, labels)):
        if isinstance(line_kwargs, (tuple, list)):
            line_kws = line_kwargs[i_win]
        for i_val, value in enumerate(make_iterable(window)):
            if i_val > 0:
                label = None
            func(value, label=str(label), **line_kws)

def slice_to_label(slice_, reduce=None):
    labels = {}
    for coord, sl in slice_.items():
        if (reduce is not None):
            reduce_str = reduce.get(coord)
            reduce_str = reduce_str if is_scalar(reduce_str) else reduce_str[0]
            reduce_str = str(reduce_str)  # TODO: map funcs to strings
        else:
            reduce_str = ''

        if utils.is_number(sl):
            labels[coord] = f'{coord}={sl:0.3g}'
        else:
            label = f'{coord}={sl}' if reduce is None else f'{reduce_str}({coord}=[{sl.start:0.3g},{sl.stop:0.3g}])'
            labels[coord] = label
    return labels

def format_poloidal_plane_ax(ax, units='m'):
    ax.set_xlabel(f'R [{units}]')
    ax.set_ylabel(f'Z [{units}]')
    ax.set_aspect('equal')
    return ax

def create_poloidal_cross_section_figure(nrow=1, ncol=1, cross_sec_axes=((0, 0),)):
    fig, axes = plt.subplots(nrow, ncol)
    if nrow==1 and ncol==1:
        format_poloidal_plane_ax(axes)
    else:
        for ax_coord in cross_sec_axes:
            ax = axes[slice(*ax_coord)]
            format_poloidal_plane_ax(ax)
    plt.tight_layout()
    return fig, axes

def plot_vessel_outline(r, z, ax=None, top=True, bottom=True, s_start_coord=(0,0), aspect='equal', ax_labels=True,
                              axes_off=False, show=True, **kwargs):
    import matplotlib.pyplot as plt
    from fire.geometry.s_coordinate import separate_rz_points_top_bottom
    if ax is None:
        fig, ax = create_poloidal_cross_section_figure(1, 1)
    else:
        fig = ax.figure

    (r_bottom, z_bottom), (r_top, z_top) = separate_rz_points_top_bottom(r, z, top_coord_start=s_start_coord,
                                                                         bottom_coord_start=s_start_coord)
    # ax.plot(r0, z0, marker='x', ls='-')
    if bottom:
        ax.plot(r_bottom, z_bottom, marker='', ls='-', **kwargs)
    if top:
        ax.plot(r_top, z_top, marker='', ls='-', **kwargs)

    ax.set_aspect(aspect)
    if axes_off:
        ax.set_axis_off()
    # elif ax_labels:  # Already done in create_poloidal_cross_section_figure()
    #     ax.set_xlabel('$R$ [m]')
    #     ax.set_ylabel('$z$ [m]')

    if show:
        plt.tight_layout()
        plt.show()
    return fig, ax

def plot_vessel_top_down(surface_radii, keys_plot, ax=None, axes_off=False, phi_labels=True, keys_plot_strong=(),
                         louvres=None):
    from matplotlib import patches
    from fire.plotting.plot_tools import get_fig_ax
    from fire.geometry.geometry import cylindrical_to_cartesian
    # TODO: Shade reference annulus, eg T4: https://stackoverflow.com/questions/22789356/plot-a-donut-with-fill-or-fill-between-use-pyplot-in-matplotlib
    fig, ax, ax_passed = get_fig_ax(ax)

    r_wall = surface_radii['R_wall']
    n_sectors = surface_radii['n_sectors']

    # Plot vessel
    wall = patches.Circle((0, 0), radius=r_wall, facecolor='b', edgecolor='k', alpha=0.1)
    ax.add_patch(wall)

    # Plot tile radii etc
    for key in keys_plot:
        r = surface_radii[key]
        alpha = 0.6 if key in keys_plot_strong else 0.3
        wall = patches.Circle((0, 0), radius=r, facecolor=None, fill=False, edgecolor='k', ls='--', alpha=alpha)
        ax.add_patch(wall)

    # Lines between sectors
    for i in np.arange(n_sectors):
        phi_sector_edge = i*360/n_sectors
        x, y = cylindrical_to_cartesian(r_wall, phi_sector_edge, angles_units='degrees')
        ax.plot([0, x], [0, y], ls='--', c='k', lw=1)

        # Lines between louvre edges
        if louvres is not None:
            for louvres_info in make_iterable(louvres): # louvres is tuple of dicts
                if isinstance(louvres_info, tuple):
                    louvres_info = dict(louvres_info)

                phi_louvre_sep = (360 / n_sectors / louvres_info['louvres_per_sector'])

                if 'radii' in louvres_info:
                    r0 = surface_radii[louvres_info['radii'][0]]
                    r1 = surface_radii[louvres_info['radii'][1]]
                else:
                    r0, r1 = 0, r_wall

                for j in np.arange(1, louvres_info['louvres_per_sector']):
                    phi = phi_sector_edge + phi_louvre_sep * j

                    x0, y0 = cylindrical_to_cartesian(r0, phi, angles_units='degrees')
                    x1, y1 = cylindrical_to_cartesian(r1, phi, angles_units='degrees')
                    ax.plot([x0, x1], [y0, y1], ls=':', c='k', lw=1)

    # Sector numbers
    for i in np.arange(n_sectors):
        x, y = cylindrical_to_cartesian(r_wall*0.9, 90-(i+0.5)*360/n_sectors, angles_units='degrees')
        ax.text(x, y, f'{i+1}', horizontalalignment='center', verticalalignment='center')

    # Phi labels
    if phi_labels:
        for i in np.arange(4):
            phi = i*360/4
            x, y = cylindrical_to_cartesian(r_wall, phi, angles_units='degrees')
            label = f'$\phi={phi:0.0f}^\circ$'
            # annotate_axis(ax, label, x, y, fontsize=16, color='k')
            ax.text(x, y, label, horizontalalignment='left', verticalalignment='bottom')
        ax_scale = 1.2
    else:
        ax_scale = 1.1

    ax.set_aspect(1)
    ax.set_xlim(-r_wall * ax_scale, r_wall * ax_scale)
    ax.set_ylim(-r_wall * ax_scale, r_wall * ax_scale)
    if axes_off:
        ax.set_axis_off()
    else:
        ax.set_xlabel(r'x [m]')
        ax.set_ylabel(r'y [m]')
        # plt.tight_layout()

    return fig, ax

# class LineBuilder:
#     def __init__(self, line):
#         self.line = line
#         self.xs = list(line.get_xdata())
#         self.ys = list(line.get_ydata())
#         self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
#
#     def __call__(self, event):
#         print('click', event)
#         if event.inaxes != self.line.axes:
#             return
#         self.xs.append(event.xdata)
#         self.ys.append(event.ydata)
#         self.line.set_data(self.xs, self.ys)
#         self.line.figure.canvas.draw()
#
# class LineDrawer(object):
#     lines = []
#     xy = []
#     def draw_line(self):
#         ax = plt.gca()
#         xy = plt.ginput(2)
#
#         x = [p[0] for p in xy]
#         y = [p[1] for p in xy]
#         line = plt.plot(x,y)
#         ax.figure.canvas.draw()
#
#         self.lines.append(line)
#         self.xy.append([x, y])

def onclick(event):
    """
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    Args:
        event:

    Returns:

    """
    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))

class CoordSelector:
    def __init__(self, fig, data=None, vars=None, slice_=None, axes=None):
        # fig = ax.figure
        # line, = ax.plot([0], [0])  # empty line

        self.fig = fig
        self.axes = make_iterable(axes) if (axes is not None) else axes
        # self.line = line
        # self.xs = list(line.get_xdata())
        # self.ys = list(line.get_ydata())

        self.data = data
        self.vars = vars
        self.slice = slice_

        # self.cid = fig.canvas.mpl_connect('button_press_event', self.left_click)
        self.cid = fig.canvas.mpl_connect('button_press_event', self)
        # self.cid = fig.canvas.mpl_connect('button_press_event', onclick)

        # logger.info(vars)

    def __call__(self, event):
    # def left_click(self, event):
    #     logger.info('click', event)
        if (self.axes is not None) and (event.inaxes not in self.axes):
            return

        coords = {}
        try:
            x, y = int(np.round(event.xdata)), int(np.round(event.ydata))
        except AttributeError as e:
            pass
        else:
            coords['pix'] = np.array([x, y])

            if (self.data is not None) and (self.vars is not None):
                # coords['data'] = []
                for var in make_iterable(self.vars):
                    try:
                        data = self.data[var]
                        value = data.sel(x_pix=x, y_pix=y)
                        if not is_scalar(value, dataarray_0d=True):
                            value = value.sel(self.slice)
                    except Exception as e:
                        pass
                    else:
                        # coords['data'].append(value)
                        try:
                            coords[var] = float(value)
                        except TypeError as e:
                            pass
                # coords['data'] = np.array(coords['data'])

            logger.info(coords)
            for key, value in coords.items():
                if key == 'pix':
                    continue
                logger.info(f'{key} = {value:0.4g}')

            # self.xs.append(event.xdata)
            # self.ys.append(event.ydata)
            # self.line.set_data(self.xs, self.ys)
            # self.line.figure.canvas.draw()

if __name__ == '__main__':
    plt.plot([1, 2, 3, 4, 5])
    ld = LineDrawer()
    ld.draw_line()  # here you click on the plot
    print(ld.lines)
    print(ld.xy)
    plt.show()
    pass