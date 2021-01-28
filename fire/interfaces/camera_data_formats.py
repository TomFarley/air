#!/usr/bin/env python

"""


Created: 
"""

import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from fire.interfaces.interfaces import read_csv
from fire import fire_paths

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

fire_root = fire_paths['root']

def read_altair_asc_image_file(path_fn):
    """Read .asc image file produced by the Altair software.

    Args:
        path_fn: Path to file

    Returns: Dataframe of image data

    """
    image = read_csv(path_fn, sep='\t', skiprows=29, header=None,
                     # skipinitialspace=True,
                     # keep_default_na=False,
                     # na_filter=False,
                     # skip_blank_lines=True
                     )
    if np.all(np.isnan(image.iloc[:, -1])):
        # Remove erroneous column due to spaces at end of lines
        image = image.iloc[:, :-1]
    return image

def read_reasearchir_csv_image_file(path_fn):

    image = read_csv(path_fn, sep=',', skiprows=31, header=None)
    return image

def read_ircam_asc_image_file(path_fn, verbose=True):

    image = read_csv(path_fn, sep='\t', skiprows=0, header=None, verbose=verbose)
    return image

def hex8_to_int(hex):
	temp = hex[-2:]+hex[4:6]+hex[2:4]+hex[0:2]
	return int(temp,16)

def hex4_to_int(hex):
	temp = hex[-2:]+hex[0:2]
	return int(temp,16)

def hex8_to_float(hex):
	import struct
	temp = hex[-2:]+hex[4:6]+hex[2:4]+hex[0:2]
	return struct.unpack('!f', bytes.fromhex(temp))[0]

def raw_to_image(raw_digital_level,width,height,digital_level_bytes):
    pixels = width*height
    # raw_digital_level_splitted = textwrap.wrap(raw_digital_level, 4)
    # iterator=map(hex4_to_int,raw_digital_level_splitted)
    # return np.flip(np.array(list(iterator)).reshape((height,width)),axis=0)
    counts_digital_level = []
    for i in range(pixels):
        counts_digital_level.append(hex4_to_int(raw_digital_level[i*digital_level_bytes:(i+1)*digital_level_bytes]))
    return np.flip(np.array(counts_digital_level).reshape((height, width)), axis=0)

def read_ircam_raw_int16_sequence_file(path_fn):

    bit_depth = 14
    # digital_level_bytes = 2
    digital_level_bytes = 4
    data_raw = open(str(path_fn), 'rb').read()
    data_hex = data_raw.hex()
    n_bytes_file = len(data_hex)
    width, height = 320, 256
    n_pixels = width * height
    bytes_per_frame = digital_level_bytes * n_pixels
    n_frames = n_bytes_file / bytes_per_frame
    if np.fmod(n_frames, 1) != 0:
        raise ValueError(f'File does not contain integer number of ({height}x{width}) frames: {n_frames}')
    n_frames = int(n_frames)

    data_movie = np.zeros((n_frames, height, width))

    print(f'Reading IRCAM raw file, {n_bytes_file} bytes, {n_frames} frames, "{path_fn}"')
    for i_frame in np.arange(n_frames):
        frame_hexdata = data_hex[i_frame*bytes_per_frame:(i_frame+1)*bytes_per_frame]

        data = raw_to_image(frame_hexdata, width, height, digital_level_bytes)
        data_movie[i_frame] = data
        # print(f'min={data.min():0.4g}, mean={data.mean():0.4g}, 1%={np.percentile(data,1):0.4g}, '
        #       f'99%={np.percentile(data,99):0.4g}, max={data.max():0.4g}')
    return data_movie

def generate_json_meta_data_file(path, fn, frame_data, meta_data_dict):
    """
    See movie_meta_required_fields in plugins_movie.py, line ~260:
      ['n_frames', 'frame_range', 't_range', 'fps', 'lens', 'exposure', 'bit_depth', 'image_shape', 'detector_window']

    Args:
        path:
        fn:
        frame_data:
        meta_data_dict:

    Returns:

    """
    from fire.interfaces.interfaces import json_dump

    fps = meta_data_dict['fps']
    period = 1/fps

    n_frames = len(frame_data)
    image_shape = list(frame_data.shape[1:])
    detector_window = [0, 0] + image_shape
    frame_numbers = np.arange(n_frames).tolist()
    frame_times = np.arange(0, n_frames*period, period).tolist()
    t_range = [min(frame_times), max(frame_times)]
    frame_range = [min(frame_numbers), max(frame_numbers)]

    dict_out = dict(n_frames=n_frames, image_shape=image_shape, detector_window=detector_window, frame_period=period,
                    lens=25e-3, bit_depth=14, t_range=t_range, frame_range=frame_range, exposure=0.25e-3,
                    frame_numbers=frame_numbers, frame_times=frame_times)
    dict_out.update(meta_data_dict)

    list_out = list(dict_out.items())

    json_dump(list_out, fn, path, overwrite=True)
    print(f'Wrote meta data file to: {path}/{fn}')

if __name__ == '__main__':
    from fire.camera.nuc import get_nuc_frame
    from fire.misc.data_structures import movie_data_to_dataarray
    from fire.plotting.image_figures import plot_movie_frames
    from fire.plotting.temporal_figures import plot_movie_data_stats

    path_fn_nuc = '/home/tfarley/repos/air/tests/test_data/lab/IRCAM_test_sequence_files/1f_test_nuc_int16.raw'
    # path_fn = '/home/tfarley/repos/air/tests/test_data/lab/IRCAM_test_sequence_files/1f_test_sequence_int16_2.raw'
    path_fn_movie = '/home/tfarley/repos/air/tests/test_data/lab/IRCAM_test_sequence_files/400f_test_sequence_int16_2.RAW'

    data_movie = read_ircam_raw_int16_sequence_file(path_fn_movie)
    data_movie = movie_data_to_dataarray(data_movie)

    # data_nuc = read_ircam_raw_int16_sequence_file(path_fn_nuc)
    data_nuc = get_nuc_frame(origin={'n': [1, 40]}, frame_data=data_movie, reduce_func='min')

    data_movie_nucsub = data_movie - data_nuc
    # data_movie_nucsub = data_movie + data_nuc
    # data_movie_nucsub = data_movie

    fn = 'movie_meta_data.json'
    path_out = '/home/tfarley/data/movies/mast_u/50002/rit/'
    generate_json_meta_data_file(path_out, fn, frame_data=data_movie, meta_data_dict={'fps': 400})

    plot_movie_frames(data_nuc, frame_label='nuc')
    plot_movie_data_stats(data_movie_nucsub)
    plot_movie_frames(data_movie_nucsub, cmap_percentiles=(2, 98)) #(0, 100))

    path = Path(fire_root) / '../tests/test_data/lab/'

    bbname = 'bb_10us.ASC'
    bgname = 'bg_10us.ASC'
    image_bg = read_ircam_asc_image_file(path / bgname)
    image_bb = read_ircam_asc_image_file(path / bbname)
    image = image_bb - image_bg
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 5))
    ax0.imshow(image_bg, cmap='gray')
    ax1.imshow(image_bb, cmap='gray')
    ax2.imshow(image, cmap='gray')
    plt.show()

    bgname = 'background_100.asc'
    bbname = 'image_100.asc'
    image_bg = read_altair_asc_image_file(path/bgname)
    image_bb = read_altair_asc_image_file(path/bbname)
    image = image_bb - image_bg
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 5))
    ax0.imshow(image_bg, cmap='gray')
    ax1.imshow(image_bb, cmap='gray')
    ax2.imshow(image, cmap='gray')
    plt.show()

    bgname = 'test_bg_filter1.csv'
    bbname = 'test_image_filter1.csv'
    image_bg = read_reasearchir_csv_image_file(path/bgname)
    image_bb = read_reasearchir_csv_image_file(path/bbname)
    image = image_bb - image_bg
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 5))
    ax0.imshow(image_bg, cmap='gray')
    ax1.imshow(image_bb, cmap='gray')
    ax2.imshow(image, cmap='gray')
    plt.show()

    pass