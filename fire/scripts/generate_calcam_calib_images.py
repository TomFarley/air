#!/usr/bin/env python

"""


Created: 
"""

import logging
from pathlib import Path

import numpy as np
import scipy as sp
import xarray as xr
import matplotlib.pyplot as plt
from skimage import exposure, data, img_as_float
from skimage.filters import rank
from skimage.morphology import disk
from skimage.io import imsave

import pyuda

from ccfepyutils.mpl_tools import get_previous_artist_color, annotate_axis
from ccfepyutils.image import hist_image_equalisation
from fire.interfaces.movie_plugins.uda import read_movie_data, read_movie_meta
from fire.image_processing import find_outlier_pixels

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# def movie_intensity_variation():


def generate_calcam_calib_images(pulse=30378, camera='rir', machine='mast', n_start=None, n_end=None, n_images=5,
                                 path_out='.'):
    # TODO: Switch to call plugin agnostic functions
    print(f'Reading movie data...')
    frame_nos, frame_times, frame_data_raw = read_movie_data(pulse, camera, n_start=n_start, n_end=n_end)
    print(f'Read data for {len(frame_nos)} frames from movie {camera}, {pulse}')
    frame_shape = np.array(frame_data_raw.shape)[1:]
    frame_rate = 1/np.mean(np.diff(frame_times))
    if True:
        # movie_meta = read_movie_meta(pulse, camera)
        print(f'Frame_rate: {frame_rate:0.1f}, image_shape: {frame_shape}')
        # print(f'Frame_rate: {movie_meta["fps"]:0.1f}, image_shape: {movie_meta["frame_shape"]}')

    dl_sat = 2**14 - 1

    frame_data_raw = xr.DataArray(frame_data_raw,
                                  coords={'t': frame_times,
                                          'x_pix': np.arange(frame_data_raw.shape[2]),
                                          'y_pix': np.arange(frame_data_raw.shape[1]),
                                          'n': ('t', frame_nos)},
                                  dims=['t', 'y_pix', 'x_pix'])

    # NUC all data
    frame_data_nuc = frame_data_raw - frame_data_raw[0]
    # use_raw = True
    use_raw = False
    if ((n_start is None) or (n_start == 0)) and (not use_raw):
        frame_data = frame_data_nuc
        nuc_str = '_nuc'
        logger.warning(f'Applied first frame NUC to all frames')
    else:
        frame_data = frame_data_raw
        nuc_str = ''
        logger.warning(f'Using raw data (no NUC)')

    path_out = Path(str(path_out).format(camera=camera, pulse=pulse))
    path_out_summaries = path_out / 'summaries'
    path_out_summaries.mkdir(exist_ok=True, parents=True)
    fn = '{camera}_{pulse}_{{n}}{nuc_str}_{{enh}}.png'.format(camera=camera, pulse=pulse, nuc_str=nuc_str)

    data = xr.Dataset(coords={'t': frame_times, 'n': ('t', frame_nos)})
    axis = (1, 2)
    data['mean'] = ('t', np.mean(frame_data.values, axis=axis))
    data['max'] = ('t', np.max(frame_data.values, axis=axis))
    data['min'] = ('t', np.min(frame_data.values, axis=axis))
    data['std'] = ('t', np.std(frame_data.values, axis=axis))
    data['range'] = ('t', np.ptp(frame_data.values, axis=axis))
    data['range'].attrs['units'] = 's'
    # Rescale std to max range
    data['std'] -= np.min(data['std'])
    data['std'] *= np.ptp(data['max'].values)/data['std']
    data['std'] += np.min(data['max'])

    df_movie_av = np.mean(data['mean'])
    i_peaks = {}
    props_peaks = {}

    fig, ax = plt.subplots()
    # vars = ['mean', 'std', 'min', 'max']
    vars = ['mean', 'max', 'std']  # full frame view
    if not np.all(data['range'] == data['max']):
        vars += ['range', 'min']  # zoomed view
    order = np.max((int(np.round(len(frame_nos) / (2*n_images))), 1))
    print(f'order={order}')
    for var in vars:
        d = data[var].swap_dims({'t': 'n'})
        d.plot.line(ax=ax, label=var, alpha=0.8)
        i_peaks[var], props_peaks[var] = sp.signal.find_peaks(data[var].values)
        # data[var][i_maxima[var]].plot(ax=ax, marker='x', ls='', color=get_previous_artist_color(ax))
        if len(i_peaks[var]) > 0:
            d[i_peaks[var]].plot.line(ax=ax, marker='x', ls='', color=get_previous_artist_color(ax))
        pass
    i_maxima_top = i_peaks['max'][np.argsort(data['max'].values[i_peaks['max']])[::-1][:n_images]]
    selected_frames = xr.DataArray(frame_data[i_maxima_top, :, :],
                                   coords={'t': frame_times[i_maxima_top],
                                           'n': ('t', frame_nos[i_maxima_top])}, dims=['t', 'x_pix', 'y_pix'])
    print(f'selected_frames: {selected_frames.coords}')
    for n in selected_frames['n']:
        ax.axvline(n, ls='--', color='gray')
    if any(data['max'] >= 0.95*dl_sat):
        ax.axhline(dl_sat, ls=':', color='gray')
    plt.legend()
    ax.set_ylabel('DL')
    plt.tight_layout()
    plt.savefig(path_out_summaries / f'movie_variation_{camera}_{pulse}.png',
                bbox_inches='tight', transparent=True, dpi=200)
    # plt.show()

    cmap = plt.cm.gray
    annotations = ['Raw', 'Bad pixels\nremoved', 'NUC', 'Contrast\nstretch 1', 'Contrast\nstretch 2',
                   'Global\nequalisation', 'Adaptive\nequalisation', 'Local\nequalisation']
    annot_fontsize = 10
    annot_pos = (0.5, 0.85)
    if np.all(frame_shape > 60):
        aspect = None
    else:
        aspect = 0.7 * frame_shape[1] / frame_shape[0]
    for i, t in enumerate(selected_frames['t'].values):
        frame = selected_frames.sel(t=t)
        frame_raw = frame_data_raw.sel(t=t)
        frame_nuc = frame_data_nuc.sel(t=t)
        # selected_frames = selected_frames.swap_dims({'t': 'n'})
        # frame = selected_frames.sel(n=79)
        n = int(frame['n'])
        print(f'Saving images for n={n}, t={t} (i={i})')
        fig, axes = plt.subplots(2, 4, sharex=True, sharey=True)
        fig.suptitle(f'{camera}, {pulse}, n={n}, t={t:0.4f} s (rank={i+1})')
        axes = axes.flatten()
        bad_pixels, frame_no_dead = find_outlier_pixels(frame_raw, tol=30, check_edges=True)
        print(f'Identified {bad_pixels.shape[1]} bad pixels')
        img = img_as_float(frame.astype(int))
        img_raw = img_as_float(frame_raw.astype(int))
        img_no_bp = img_as_float(frame_no_dead.astype(int))
        img_nuc = img_as_float(frame_nuc.astype(int))

        # Raw image
        i_ax = 0
        ax = axes[i_ax]
        ax.imshow(frame_raw, cmap=cmap, aspect=aspect)
        for y, x in zip(bad_pixels[0], bad_pixels[1]):
            ax.plot(x, y, 'ro', mfc='none', mec='r', ms=8)
        annotate_axis(ax, annotations[i_ax], x=annot_pos[0], y=annot_pos[1], fontsize=annot_fontsize)

        # Bad pix removed
        i_ax += 1
        ax = axes[i_ax]
        ax.imshow(img_no_bp, cmap=cmap, aspect=aspect)
        annotate_axis(ax, annotations[i_ax], x=annot_pos[0], y=annot_pos[1], fontsize=annot_fontsize)

        # NUC
        i_ax += 1
        ax = axes[i_ax]
        # if (n_start is None) or (n_start == 0):
        #     img = img_as_float(img_nuc.astype(int))
        #     logger.warning(f'Using NUC frame as input for enhancements')
        ax.imshow(frame_nuc, cmap=cmap, aspect=aspect)
        annotate_axis(ax, annotations[i_ax], x=annot_pos[0], y=annot_pos[1], fontsize=annot_fontsize)

        # Contrast stretching 1
        i_ax += 1
        ax = axes[i_ax]
        p0, p1 = np.percentile(img_no_bp, (2, 98))
        img_rescale_1 = exposure.rescale_intensity(img, in_range=(p0, p1))
        ax.imshow(img_rescale_1, cmap=cmap, aspect=aspect)
        annotate_axis(ax, annotations[i_ax], x=annot_pos[0], y=annot_pos[1], fontsize=annot_fontsize)

        # Contrast stretching 2
        i_ax += 1
        ax = axes[i_ax]
        p0, p1 = np.percentile(img_no_bp, (20, 98))
        img_rescale_2 = exposure.rescale_intensity(img, in_range=(p0, p1))
        ax.imshow(img_rescale_2, cmap=cmap, aspect=aspect)
        annotate_axis(ax, annotations[i_ax], x=annot_pos[0], y=annot_pos[1], fontsize=annot_fontsize)

        # Global equalization
        i_ax += 1
        ax = axes[i_ax]
        img_eq = exposure.equalize_hist(img)
        ax.imshow(img_eq, cmap=cmap, aspect=aspect)
        annotate_axis(ax, annotations[i_ax], x=annot_pos[0], y=annot_pos[1], fontsize=annot_fontsize)

        # Adaptive Equalization
        try:
            i_ax += 1
            ax = axes[i_ax]
            # img_adapteq = exposure.equalize_adapthist(img, kernel_size=np.array(img.shape)/12, clip_limit=0.03)
            img_adapteq = hist_image_equalisation(img, image_equalisation_adaptive=True, clip_limit=2,
                                                  tile_grid_size=(8, 8))
            imsave(path_out / fn.format(n=n, enh='eq_adapt'), img_adapteq)
            ax.imshow(img_adapteq, cmap=cmap, aspect=aspect)
            annotate_axis(ax, annotations[i_ax], x=annot_pos[0], y=annot_pos[1], fontsize=annot_fontsize)
        except Exception as e:
            pass

        # Local equalisation
        i_ax += 1
        ax = axes[i_ax]
        img_loc = rank.equalize(img, disk(20))
        imsave(path_out / fn.format(n=n, enh='eq_loc'), img_loc)
        ax.imshow(img_loc, cmap=cmap, aspect=aspect)
        annotate_axis(ax, annotations[i_ax], x=annot_pos[0], y=annot_pos[1], fontsize=annot_fontsize)

        imsave(path_out / fn.format(n=n, enh='rescale_1'), img_rescale_1)
        imsave(path_out / fn.format(n=n, enh='rescale_2'), img_rescale_2)
        imsave(path_out / fn.format(n=n, enh='eq_glob'), img_eq)

        for ax in axes:
            ax.set_axis_off()
        fig.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.savefig(path_out_summaries / fn.format(n=n, enh='summary'),
                    bbox_inches='tight', transparent=True, dpi=300)
        # plt.show()
        pass
    plt.show()

if __name__ == '__main__':
    # pulse = 30378
    # pulse = 28866  # Low power
    # pulse = 29210  # High power
    # pulse = 30378  # High ELM surface temperatures ~450 C

    # Full frame air:
    # pulse = 23586  # Full frame with clear spatial calibration - ref
    pulse = 28911
    # pulse = 29936
    # pulse = 30458
    # pulse = 29934
    # pulse = 30009

    # Full frame ait:
    # pulse = 30459
    # pulse = 30458


    camera = 'rir'
    # camera = 'rit'
    n_start = None
    n_end = None
    # n_start = 3200
    # n_end = 3500
    # n_start = 0
    # n_end = 110
    # n_images = 4
    n_images = 15
    # path_out = Path('./calibration_images/{camera}/{pulse}')
    path_out = Path('~/calcam2/input_images/{camera}/{pulse}').expanduser()
    generate_calcam_calib_images(pulse=pulse, camera=camera, machine='mast', n_start=n_start, n_end=n_end,
                                 path_out=path_out, n_images=n_images)
    pass