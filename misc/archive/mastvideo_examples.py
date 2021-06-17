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

def download_ipx_via_http_request(camera, pulse, path_out='~/data/movies/{machine}/{pulse}/',
                                  fn_out='{camera}0{pulse}.ipx', verbose=True):
    import requests
    from fire.interfaces.io_basic import mkdir

    machine = 'mast_u' if (pulse > 40000) else 'mast'

    path_out = Path(path_out.format(camera=camera, pulse=pulse, machine=machine)).expanduser()
    fn_out = fn_out.format(camera=camera, pulse=pulse, machine=machine)
    path_fn = path_out / fn_out
    mkdir(path_out, depth=2)

    url = f'http://video-replay-dev.mastu.apps.l/0{pulse}/{camera}/raw'
    r = requests.get(url, allow_redirects=True)

    if b'404 Not Found' in r.content:
        message = f'Failed to write ipx file to "{path_fn}" from {url}. URL not found.'
        logger.warning(message)
    else:
        open(path_fn, 'wb').write(r.content)
        message = f'Wrote ipx file to "{path_fn}" from {url}'

    logger.debug(message)
    if verbose:
        print(message)

    return fn_out

if __name__ == '__main__':
    # pulse = 27880
    pulse = 43651
    # pulse = 30378
    # pulse = 28000

    camera = 'rit'
    # camera = 'rgb'
    # camera = 'rir'

    machine = 'mast_u'

    # download_ipx_via_http_request(camera, pulse)
    # read_ipx_example('rgb', 30378)
    read_ipx_example(camera, pulse, machine=machine)
    write_ipx_example()
    pass