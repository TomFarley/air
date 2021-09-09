#!/usr/bin/env python

"""


Created: 
"""

import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger.propagate = False

def read_ipx_example(camera, pulse, machine='mast', verbose=True):
    from mastvideo import load_ipx_file, VideoDecoder

    path_fn = f'/home/tfarley/data/movies/{machine}/{pulse}/{camera}/{camera}0{pulse}.ipx'

    # open the file
    ipx = load_ipx_file(open(path_fn, mode='rb'))

    # display information about the file
    print(ipx.header)
    print(ipx.sensor)

    video = VideoDecoder(ipx)

    frames = np.zeros((ipx.header.num_frames, ipx.header.frame_height,  ipx.header.frame_width))
    times = np.array([frame.time for frame in ipx.frames])

    # iterate over frames saving each one into a seperate PNG file
    for i, frame in enumerate(video.frames()):
        frames[i] = np.array(frame)
        # frame.save(f'{camera}0{pulse}_frame-{i:04}.png')

    message = f'Read ipx file: "{path_fn}"'
    logger.debug(message)
    if verbose:
        print(message)

    # print(frames)

    frame = frames[int(3*len(frames)/4)]
    # frame *= 1 / np.quantile(frame, 0.99)
    # frame = -np.expm1(-frame)
    plt.imshow(frame)
    plt.show()

    return ipx, video, frames, times

def write_ipx_example(verbose=True):
    from mastvideo import write_ipx_file, IpxHeader, IpxSensor, SensorType, ImageEncoding

    ipx, video, frames, times = read_ipx_example('rir', 30378)
    nframes, height, width = tuple(frames.shape)
    frames = list(video.frames())  # PIL images
    frames = [frame.convert('I;16') for frame in frames]

    # fill in some dummy fields
    header = IpxHeader(
        shot=12345,
        date_time='2008-05-07T14:20:40',

        camera='camera name',
        view='view name',
        lens='lens name',
        trigger=-0.01,
        exposure=300,

        num_frames=len(frames),
        frame_width=width,
        frame_height=height,
        depth=14,
    )

    sensor = IpxSensor(
        type=SensorType.MONO,
    )

    fn = 'test012345.ipx'

    with write_ipx_file(
            fn, header, sensor, version=1,
            encoding=ImageEncoding.JPEG2K,
    ) as ipx:
        # write out the frames
        for time, frame in zip(times, frames):
            ipx.write_frame(time, frame.convert('I;16'))

    message = f'Wrote test ipx file: "{fn}"'
    logger.debug(message)
    if verbose:
        print(message)


def rewrite_ipx_file_with_mastvideo(pulse, camera='rir'):
    from fire.plugins.movie_plugins.ipx import (read_movie_data_with_mastmovie, read_movie_data_with_mastmovie,
        read_movie_data, read_movie_meta, write_ipx_with_mastmovie)

    n = 350

    fn_ipx_in = Path(f'/net/fuslsc.mast.l/data/MAST_IMAGES/0{str(pulse)[:2]}/{pulse}/{camera}0{pulse}.ipx')
    fn_ipx_out = f'{camera}0{pulse}_test.ipx'

    print(f'In: {fn_ipx_in}')
    print(f'Out: {Path(fn_ipx_out).resolve()}')

    frame_numbers, frame_times, data_original = read_movie_data(fn_ipx_in)
    meta_original = read_movie_meta(fn_ipx_in)
    meta_original['frame_times'] = frame_times
    meta_original['shot'] = pulse

    pil_frames = write_ipx_with_mastmovie(fn_ipx_out, data_original, header_dict=meta_original, apply_nuc=False)

    frame_numbers, frame_times, data_new = read_movie_data(fn_ipx_out)
    meta_new = read_movie_meta(fn_ipx_out)

    data_diff = data_new - data_original
    data_diff_max_frames = np.max(data_diff, axis=(1,2))
    n_worst = np.argmax(data_diff_max_frames)

    frame_original = data_original[n_worst]
    frame_new = data_new[n_worst]

    print(meta_original)
    print(meta_new)
    print('Original:')
    print(frame_original[200:250, 200:250])
    print('New:')
    print(frame_new[200:250, 200:250])
    print(data_diff_max_frames)
    print(np.sort(data_diff_max_frames)[-10:])
    print(f'Max movie diff = {np.max(data_diff)}')
    print(f'n_worst = {n_worst}')

    # plt.ion()
    fig, (ax0, ax1) = plt.subplots(1, 2, sharex=True, sharey=True)
    fig.suptitle(f'{pulse}, n={n_worst}')
    ax0.imshow(frame_original, interpolation='none', origin='upper', cmap='gray', vmin=frame_original.min(),
               vmax=2**14-1)
    ax0.set_title(f'Original')
    im2 = ax1.imshow(frame_new, interpolation='none', origin='upper', cmap='gray', vmin=frame_new.min(), vmax=2**14-1)
    ax1.set_title(f'Mastvideo output')
    plt.colorbar(im2)
    plt.tight_layout()
    plt.draw()
    plt.show()

    pil_frames[n].show(title='PIL native show')
    pass

if __name__ == '__main__':

    # pulse = 14997
    pulse = 27880
    # pulse = 43651
    # pulse = 30378
    # pulse = 28000

    camera = 'rit'
    # camera = 'rgb'
    # camera = 'rir'

    machine = 'mast_u'

    rewrite_ipx_file_with_mastvideo(pulse, camera)

    # download_ipx_via_http_request(camera, pulse)
    # read_ipx_example('rgb', 30378)
    # read_ipx_example(camera, pulse, machine=machine)
    # write_ipx_example()
    pass