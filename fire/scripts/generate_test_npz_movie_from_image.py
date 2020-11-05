#!/usr/bin/env python

"""


Created: 
"""

import logging, datetime
from typing import Union, Iterable, Sequence, Tuple, Optional, Any, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from fire.plugins.movie_plugins import imstack
from fire.plotting.image_figures import animate_image_data
from fire.misc.utils import mkdir

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

def generate_test_npz_movie_from_cad_image(fn_out, path_in, nframes=20, dt=1e-4):
    fn_out = Path(fn_out).resolve()

    movie_meta = imstack.read_movie_meta(path_in)
    frame_nos, frame_times, frame = imstack.read_movie_data(path_in)
    frame = frame[0]

    frame_data = np.zeros((nframes,)+frame.shape)
    frame_nos = np.arange(0, nframes)
    frame_times = np.arange(0, nframes*dt, dt)
    bit_depth = 14
    max_value = 2**(bit_depth-1)
    expsoure = 232e-6
    lens = 13e-2  # 13e-3
    image_shape = np.array(frame_data.shape[1:])
    detector_window = np.array([0, 0, *image_shape[::-1]])

    # Raise minimum intensity well above zero
    if np.min(frame) < 0.1 * np.max(frame):
        frame = frame + 0.3*np.max(frame)
    # Normalise frame intensity to fraction of bit depth
    fraction_of_bitdepth = 0.55
    frame = fraction_of_bitdepth * max_value * frame / np.max(frame)

    # Add inter-frame variation
    scale_factors = 1 + 0.25 * np.sin(1.5*np.pi + 1.5*np.pi*frame_nos/len(frame_nos))
    scale_factors /= np.max(scale_factors)

    if False:
        fig, ax = plt.subplots()
        ax.plot(scale_factors)
        plt.show()
    frame_data = scale_factors[:, np.newaxis, np.newaxis] * frame

    data = dict(bit_depth=bit_depth, exposure=expsoure, lens=lens,
                imstack_filenames=movie_meta['imstack_header']['imstack_filenames'],
                n_frames=len(frame_nos), frame_range=[np.min(frame_nos), np.max(frame_nos)],
                t_range=[np.min(frame_times), np.max(frame_times)], image_shape=image_shape,
                detector_window=detector_window, fps=1/dt,
                author='tfarley', date_created=str(datetime.datetime.today()))

    np.savez(fn_out, frames=frame_data, time=frame_times, frame_nos=frame_nos, scale_factors=scale_factors, **data)
    print(f'Wrote npz test movie to {fn_out} using calibration data with meta data: \n{data}')

def generate_npz_movie_from_calib_asc_images(fn_out, path_in, exposure_time=30, dt=1e-4, plot=True):
    from fire.interfaces.camera_data_formats import read_ircam_asc_image_file

    fn_out = Path(fn_out).resolve()

    mkdir(fn_out, depth=1, verbose=True)
    fn_bb = f'bb_{exposure_time}us.ASC'
    fn_nuc = f'bg_{exposure_time}us.ASC'

    path_in = Path(path_in)
    temperature_dirs = np.array([item for item in path_in.iterdir() if item.is_dir()])
    temperatures = np.array([int(path.name) for path in temperature_dirs])
    i_sort = np.argsort(temperatures)
    temperatures = temperatures[i_sort]
    temperature_dirs = temperature_dirs[i_sort]
    temperatures = np.concatenate([[23], temperatures])

    nframes = len(temperature_dirs) + 1
    bit_depth = 14
    max_value = 2 ** (bit_depth - 1)
    expsoure = exposure_time*1e-6
    lens = 50e-2  # 13e-3

    # Get NUC frame for first temperature
    fn_nuc = temperature_dirs[0]/fn_nuc
    frame_nuc = read_ircam_asc_image_file(fn_nuc).T
    frame_data = np.zeros((nframes,) + frame_nuc.shape)
    frame_data[0] = frame_nuc

    image_shape = np.array(frame_data.shape[1:])
    detector_window =  np.array([0, 0, *image_shape[::-1]])

    for i, temp_dir in enumerate(temperature_dirs, start=1):
        path_bb = temp_dir / fn_bb
        frame = read_ircam_asc_image_file(path_bb, verbose=True).T
        frame_data[i] = frame

    # frame_data = frame_data.T  # Transpose to match test calcam calibration

    frame_nos = np.arange(0, nframes)
    frame_times = np.arange(0, nframes*dt, dt)

    if plot:
        for n in np.arange(len(frame_data)):
            plt.imshow(frame_data[n])
            plt.show()

    data = dict(bit_depth=bit_depth, exposure=expsoure, lens=lens, temperatures=temperatures,
                calib_files_path=str(path_in),
                n_frames=len(frame_nos), frame_range=[np.min(frame_nos), np.max(frame_nos)],
                t_range=[np.min(frame_times), np.max(frame_times)], image_shape=image_shape,
                detector_window=detector_window, fps=1/dt)

    np.savez(fn_out, frames=frame_data, time=frame_times, frame_nos=frame_nos, **data)
    print(f'Wrote npz test movie to {fn_out} with meta data: \n{data}')

if __name__ == '__main__':

    if True:
        pulse = 50000
        path_in = f'/home/tfarley/data/movies/mast_u/{pulse}/rit/images/'
        fn_out = path_in + '../' + f'rit_{pulse}.npz'
        generate_test_npz_movie_from_cad_image(fn_out, path_in, nframes=20)

    print()
    if True:
        pulse = 50001
        exposure_time=30
        path_in = '/home/tfarley/repos/ir_tools/calibration/AT_IDL_tools/IR_ENH_calib/IRcam_0101_50mm_no_ND_20190219/'
        # path_in = '/home/tfarley/repos/ir_tools/calibration/AT_IDL_tools/IR_ENH_calib/IRcam_0101_50mm_20181121/'
        fn_out = f'/home/tfarley/data/movies/mast_u/{pulse}/rit/' + f'rit_{pulse}.npz'
        generate_npz_movie_from_calib_asc_images(fn_out=fn_out, path_in=path_in, exposure_time=exposure_time,
                                                 plot=False)


    pass