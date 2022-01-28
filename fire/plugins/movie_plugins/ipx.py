# -*- coding: future_fstrings -*-
#!/usr/bin/env python

"""
This module defines functions for interfacing with MAST (2000-2013) data archives and systems.
"""

import logging
from typing import Union, Iterable, Tuple, Optional
from pathlib import Path
import numbers
from copy import copy
from collections import namedtuple

import numpy as np
from fire.plugins.movie_plugins.ipx_standard import (check_ipx_detector_window_meta_data,
    get_detector_window_from_ipx_header, convert_ipx_header_to_uda_conventions)

try:
    from fire.plugins.movie_plugins import ipx_standard
    IPX_HEADER_FIELDS = ipx_standard.UDA_IPX_HEADER_FIELDS
except ImportError as e:
    IPX_HEADER_FIELDS = ('left', 'top', 'width', 'height', 'right', 'bottom')

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

MovieData = namedtuple('movie_plugin_frame_data', ['frame_numbers', 'frame_times', 'frame_data'])

movie_plugin_name = 'ipx'
plugin_info = {'description': 'This plugin reads IPX1/2 format MAST movie files'}

def get_freia_ipx_path(pulse, diag_tag_raw):
    """Return path to ipx file on UKAEA freia cluster

    :param pulse: Shot/pulse number or string name for synthetic movie data
    :param camera: Name of camera to analyse (unique name of camera or diagnostic code)
    :return: Path to ipx files
    """
    pulse = str(pulse)
    # MAST_IMAGES = '/net/fuslsa/data/MAST_IMAGES/'
    MAST_IMAGES = '/net/fuslsc.mast.l/data/MAST_IMAGES/'
    ipx_path_fn = MAST_IMAGES + f"0{pulse[0:2]}/{pulse}/{diag_tag_raw}0{pulse}.ipx"
    return ipx_path_fn

def read_movie_meta_with_pyipx(path_fn: Union[str, Path], transforms: Iterable[str]=()) -> dict:
    """Read frame data from MAST IPX movie file format.

    :param path_fn: Path to IPX movie file
    :type path_fn: str, Path
    :param transforms: List of of strings describing transformations to apply to frame data. Options are:
                        'reverse_x', 'reverse_y', 'transpose'
    :type transforms: list
    :return: Dictionary of ipx file information
    :type: dict
    """
    from pyIpx.movieReader import ipxReader

    if not Path(path_fn).is_file():
        raise FileNotFoundError(f'IPX file does not exist: {path_fn}')
    # Read file header and first frame
    vid = ipxReader(filename=path_fn)
    ipx_header = vid.file_header
    ipx_header = convert_ipx_header_to_uda_conventions(ipx_header)
    ret, frame0, frame_header0 = vid.read(transforms=transforms)
    n_frames = ipx_header.get('n_frames', ipx_header.get('numFrames'))
    i_last_frame = n_frames - 1

    # Last frame doesn't always load, so work backwards to last successfully loaded frame
    ret = False
    while not ret:
        # Read last frame
        vid.set_frame_number(i_last_frame)
        ret, frame_end, frame_header_end = vid.read(transforms=transforms)
        if not ret:
            # File closes when it fails to load a frame, so re-open
            vid = ipxReader(filename=path_fn)
            i_last_frame -= 1
    vid.release()

    # Collect summary of ipx file meta data
    # file_header['ipx_version'] = vid.ipx_type
    movie_meta = {'movie_format': '.ipx'}
    movie_meta['n_frames'] = ipx_header['n_frames']
    movie_meta['frame_range'] = np.array([0, i_last_frame])
    movie_meta['t_range'] = np.array([float(frame_header0['time_stamp']), float(frame_header_end['time_stamp'])])
    movie_meta['t_before_pulse'] = np.abs(ipx_header.get('trigger', movie_meta['t_range'][0]))
    movie_meta['image_shape'] = np.array(frame0.shape)
    movie_meta['fps'] = (i_last_frame) / np.ptp(movie_meta['t_range'])
    movie_meta['exposure'] = ipx_header['exposure']
    movie_meta['bit_depth'] = ipx_header['depth']
    movie_meta['lens'] = ipx_header['lens'] if 'lens' in ipx_header else 'Unknown'

    # Make sure detector window fields are extracted from ipx header
    for key in IPX_HEADER_FIELDS:
        if key in ipx_header:
            movie_meta[key] = ipx_header[key]

    # TODO: Add filter name?

    # TODO: Move derived fields to common function for all movie plugins: image_shape, fps, t_range
    # TODO: Check ipx field 'top' follows image/calcam conventions
    check_ipx_detector_window_meta_data(movie_meta, plugin='ipx', fn=path_fn, modify_inplace=True)  # Complete missing fields
    movie_meta['detector_window'] = get_detector_window_from_ipx_header(movie_meta)  # left, top, width, height
    movie_meta['ipx_header'] = ipx_header
    return movie_meta

def read_movie_data_with_pyipx(path_fn: Union[str, Path],
                               n_start:Optional[int]=None, n_end:Optional[int]=None, stride:Optional[int]=1,
                               frame_numbers: Optional[Union[Iterable, int]]=None,
                               transforms: Optional[Iterable[str]]=()) -> MovieData:
    """Read frame data from MAST IPX movie file format.

    :param path_fn: Path to IPX movie file
    :type path_fn: str, Path
    :param frame_numbers: Frame numbers to read (should be monotonically increasing)
    :type frame_numbers: Iterable[int]
    :param transforms: List of of strings describing transformations to apply to frame data. Options are:
                        'reverse_x', 'reverse_y', 'transpose'
    :type transforms: Optional[Iterable[str]]
    :return: frame_nos, times, data_frames,
    :type: (np.array, np.array ,np.ndarray)
    """
    from pyIpx.movieReader import ipxReader

    path_fn = Path(path_fn)
    if not path_fn.exists():
        raise FileNotFoundError(f'Ipx file not found: {path_fn}')
    if transforms is None:
        transforms = ()
    vid = ipxReader(filename=path_fn)
    ipx_header = vid.file_header
    n_frames_movie = ipx_header['numFrames']
    if frame_numbers is None:
        if (n_start is not None) and (n_end is not None):
            frame_numbers = np.arange(n_start, n_end + 1, stride, dtype=int)
        else:
            frame_numbers = np.arange(n_frames_movie, dtype=int)
    elif isinstance(frame_numbers, numbers.Number):
        frame_numbers = np.array([frame_numbers])
    else:
        frame_numbers = np.array(frame_numbers)
    frame_numbers[frame_numbers < 0] += n_frames_movie
    if any((frame_numbers >= n_frames_movie)):
        raise ValueError(f'Requested frame numbers outside of movie range: '
                         f'{frame_nos[(frame_nos >= vid.file_header["numFrames"])]}')
    if any(np.fmod(frame_numbers, 1) > 1e-5):
        raise ValueError(f'Fractional frame numbers requested from ipx file: {frame_numbers}')
    # Allocate memory for frames
    frame_data = np.zeros((len(frame_numbers), ipx_header['height'], ipx_header['width']), dtype=np.uint16)
    frame_times = np.zeros_like(frame_numbers, dtype=float)

    # To efficiently read the video the frames should be loaded in monotonically increasing order
    frame_numbers = np.sort(frame_numbers).astype(int)
    n, n_end = frame_numbers[0], frame_numbers[-1]

    i_data = 0
    vid.set_frame_number(n)
    while n <= n_end:
        if n in frame_numbers:
            # frames are read with 16 bit dynamic range, but values are 14 bit!
            try:
                ret, frame, header = vid.read(transforms=transforms)
            except Exception as e:
                raise e
            if ret:
                frame_data[i_data, :, :] = frame
                frame_times[i_data] = header['time_stamp']
                i_data += 1
            else:
                logger.warning(f'Failed to read frame {n}/{n_end} from {path_fn}')
        elif n > n_frames_movie:
            logger.warning('n={} outside ipx movie frame range'.format(n))
            break
        else:
            # Increment vid frame number without reading data
            vid._skip_frame()
        n += 1
    vid.release()

    frame_data = frame_data.astype(np.uint16)

    return MovieData(frame_numbers, frame_times, frame_data)

def read_movie_meta_with_mastmovie(path_fn: Union[str, Path], transforms: Iterable[str]=()) -> dict:
    """Read frame data from MAST IPX movie file format.

    :param path_fn: Path to IPX movie file
    :type path_fn: str, Path
    :param transforms: List of of strings describing transformations to apply to frame data. Options are:
                        'reverse_x', 'reverse_y', 'transpose'
    :type transforms: list
    :return: Dictionary of ipx file information
    :type: dict
    """
    from mastvideo import load_ipx_file

    if not Path(path_fn).is_file():
        raise FileNotFoundError(f'IPX file does not exist: {path_fn}')

    # open the file
    ipx = load_ipx_file(open(path_fn, mode='rb'))
    ipx.header.validate()

    # display information about the file
    # print(ipx.header)
    # print(ipx.sensor)
    ref_frames = [np.array(ipx.decode_frame(ref.data)).astype(np.uint16) for ref in ipx.ref_frames]
    if ipx.badpixels is not None:
        bad_pixels = np.array(ipx.decode_frame(ipx.badpixels.data)).astype(np.uint16)
    else:
        bad_pixels = None

    movie_meta = dict(movie_format='.ipx')

    header_fields = {  # mastvideo name to ipx/uda convention
                     'num_frames': 'n_frames',
                     'exposure': 'exposure',
                     'depth': 'bit_depth',
                     'lens': 'lens',
                     'board_temperature': 'board_temp',
                     # 'bytes_per_decoded_frame': 'bytes_per_decoded_frame',
                     'camera': 'camera',
                     # 'count': 'count',
                     'date_time': 'date_time',
                     'filter': 'filter',
                     'frame_height': 'height',
                     'frame_width': 'width',
                     # 'index': 'index',
                     'orientation': 'orientation',
                     # 'pil_image_mode': 'pil_image_mode',
                     # 'pixels_per_frame': 'pixels_per_frame',
                     'pre_exposure': 'pre_exp',
                     'sensor_temperature': 'ccd_temp',
                     'shot': 'shot',
                     'strobe': 'strobe',
                     'trigger': 'trigger',
                     'view': 'view',
                    # 'codex': 'codec',
                    # 'file_format': 'ID'
    }

    movie_meta.update({name: getattr(ipx.header, key) for key, name in header_fields.items()})

    for i, ref_frame in enumerate(ref_frames):
        movie_meta[f'ref_frame_{i}'] = ref_frame
        if i == 0:
            movie_meta[f'nuc_frame'] = ref_frame
        elif i == 1:
            movie_meta[f'nuc_frame_2'] = ref_frame

    movie_meta['bad_pixels_frame'] = bad_pixels

    sensor_fields = {
                     'binning_h': 'hbin',
                     'binning_v': 'vbin',
                     # 'count': 'count',
                     'sensor_gain': 'gain',
                     # 'index': 'index',
                     'offset': 'offset',
                     'taps': 'taps',
                     # 'type': 'sensor_type',
                     'window_bottom': 'bottom',
                     'window_left': 'left',
                     'window_right': 'right',
                     'window_top': 'top',
                     # 'is_color': 'color'
                      }
    # TODO: Rename meta data window fields to window_<> for all plugins/pass through reformat function
    movie_meta.update({name: getattr(ipx.sensor, key, None) for key, name in sensor_fields.items()})
    # movie_meta['sensor_type'] = movie_meta['sensor_type'].value  # Keep string not class

    movie_meta['frame_range'] = np.array([0, movie_meta['n_frames']-1])
    movie_meta['t_range'] = np.array([ipx.frames[0].time, ipx.frames[-1].time])  # TODO: Refine
    movie_meta['image_shape'] = np.array([movie_meta['height'], movie_meta['width']])
    movie_meta['fps'] = movie_meta['n_frames'] / (movie_meta['t_range'][1] - movie_meta['t_range'][0])

    check_ipx_detector_window_meta_data(movie_meta, plugin='raw', fn=path_fn, modify_inplace=True)  # Complete missing fields
    movie_meta['detector_window'] = get_detector_window_from_ipx_header(movie_meta)  # left, top, width, height

    frame_times = np.array([frame.time for frame in ipx.frames])
    movie_meta['fps'] = 1 / np.median(np.diff(frame_times))
    # movie_meta['fps'] = (video.n_frames - 1) / np.ptp(times)  # sucestible to errors in start/end frame times

    # raise NotImplementedError
    return movie_meta


def read_movie_data_with_mastmovie(path_fn: Union[str, Path],
                                   n_start: Optional[int] = None, n_end: Optional[int] = None,
                                   stride: Optional[int] = 1,
                                   frame_numbers: Optional[Union[Iterable, int]] = None,
                                   transforms: Optional[Iterable[str]] = (),
                                   verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from mastvideo import load_ipx_file
    from fire.misc.utils import make_iterable

    if not Path(path_fn).is_file():
        raise FileNotFoundError(f'IPX file does not exist: {path_fn}')

    # open the file
    ipx = load_ipx_file(open(path_fn, mode='rb'))

    # video = VideoDecoder(ipx)  # convert to 8bit (interpolate to RGB, if valid for sensor) for conversion to mpeg video
    # frames = list(video.frames())

    bit_depth = ipx.header.depth
    n_frames_movie = ipx.header.num_frames

    frame_times_all = np.array([frame.time for frame in ipx.frames])

    if frame_numbers is None:
        n_start = 0 if (n_start is None) else n_start
        n_end = n_frames_movie-1 if (n_end is None) else n_end
        frame_numbers = np.arange(n_start, n_end+1, stride)
    else:
        frame_numbers = make_iterable(frame_numbers, ndarray=True)

    n_frames_read = len(frame_numbers)

    if any((frame_numbers >= n_frames_movie)):
        raise ValueError(f'Requested frame numbers outside of movie range: '
                         f'{frame_nos[(frame_nos >= vid.file_header["numFrames"])]}')
    if any(np.fmod(frame_numbers, 1) > 1e-5):
        raise ValueError(f'Fractional frame numbers requested from ipx file: {frame_numbers}')

    frame_numbers = frame_numbers.astype(int)
    frame_times_all = frame_times_all[frame_numbers]

    frame_data = np.zeros((n_frames_read, ipx.header.frame_height, ipx.header.frame_width))

    # TODO: Deal with non n=0 first frame?
    # iterate over frames converting to numpy array
    i = 0
    for n, frame in enumerate(ipx.frames):
        if n in frame_numbers:
            image = ipx.decode_frame(frame.data)  # 'I;16'
            frame_data[i] = np.array(image).astype(np.uint16)
            i += 1
    if i != n_frames_read:
        raise AssertionError(f"i != n_frames: {i} != {n_frames_read}")

    # IPX files need to be written and read with x4 factor applied
    pil_depth_correction = 2 ** (16 - bit_depth)
    frame_data /= pil_depth_correction

    message = f'Read ipx file with mastmovie: "{path_fn}"'
    logger.debug(message)
    if verbose:
        print(message)

    return frame_numbers, frame_times_all, frame_data

def write_ipx_with_mastmovie(path_fn_ipx: Union[Path, str], movie_data: np.ndarray, header_dict: dict,
                             apply_nuc=False, create_path=False, verbose: bool=True):
    from PIL import Image
    from mastvideo import write_ipx_file, IpxHeader, IpxSensor, SensorType, ImageEncoding
    from fire.plugins.machine_plugins import mast_u
    from fire.interfaces import interfaces

    movie_data = movie_data.astype(np.uint16)

    n_frames, height, width = tuple(movie_data.shape)
    image_shape = (height, width)
    left, top, right, bottom = np.array([np.abs(header_dict[key]) if key in header_dict else None
                                                        for key in ('left', 'top', 'right', 'bottom')])

    pulse = int(header_dict.get('shot', header_dict.get('pulse')))
    camera =     header_dict.get('camera', 'IRCAM Velox_81kL_0102A18CH_FAST')
    # fps =     header_dict['fps']
    exposure =     int(header_dict.get('exposure', 0.25e-3)* 1e6)
    lens =     str(header_dict.get('lens', 25e-3))
    view =     header_dict.get('view', 'HL04_A-tangential')
    t_before_pulse =     header_dict['t_before_pulse']  # 1e-1
    # period =     header_dict.get('frame_period', 1/fps)
    orient =     int(header_dict.get('orient', 90))  # Rotation to apply to data to get correct orientation
    filter =     str(header_dict.get('filter', 'None'))  # nd filter etc
    taps =     int(header_dict.get('taps', 1))  # nd filter etc
    depth = int(header_dict.get('bit_depth', header_dict.get('depth', 14)))
    date_time =     header_dict.get('date_time', mast_u.get_shot_date_time(pulse))
    taps =     header_dict.get('taps', None)

    times = header_dict['frame_times']

    ipx_version = int(header_dict.get('ID', '1')[-1])

    shot = interfaces.digest_shot_file_name(fn=path_fn_ipx)['shot']
    datetime = mast_u.get_shot_date_time(shot)

    trigger = header_dict.get('t_before_pulse', header_dict.get('trigger',
                                                                header_dict.get('ipx_header', {}).get('trigger')))
    trigger=-np.abs(trigger)

    # TODO: Use pycpf when it working to get datetime eg '2013-06-11T14:27:21'
    header_dict_subset = dict(shot=pulse, date_time=datetime,
                              camera=camera, view=view, lens=lens, filter=filter,
                              trigger=trigger, exposure=exposure,
                              num_frames=n_frames, depth=depth,
                              frame_width=width, frame_height=height, orientation=orient,
                              # orientation=0, pre_exposure=0,
                              # board_temperature=0, sensor_temperature=0, strobe=0,
                        # Other fields are in Sensor object
                              # codec='JP2',
                              # file_format='IPX 01',
                              # offset=0.0, ccdtemp=-1, numFrames, filter, codex='JP2', ID='IPX 01'
                              )

    sensor_dict = dict(
        window_left=left, window_right=right, window_top=top, window_bottom=bottom,
        binning_h=0, binning_v=0,  # 0 = no binning
        taps=taps,   # taps=Number of digitizer channels
        gain=None, offset=None)

    if image_shape == (256, 320):
        # header_dict_subset['top'] = 0
        # header_dict_subset['bottom'] = 0  # TODO: check which to set to zero from MAST examples
        # header_dict_subset['left'] = 0  # TODO: Ask Sam about updating permissable fields to include 'left'?
        pass
    else:
        # raise NotImplementedError(f'Detector subwindowed: {image_shape}')
        print(f'Detector subwindowed? image_shape={image_shape}')

    # header_dict = complete_meta_data_dict(header_dict, n_frames=n_frames, image_shape=image_shape)
    # header_dict = {**header_dict_default, **header_dict}
    #
    # times = header_dict['frame_times']
    # frames = [Image.fromarray(frame, mode='I;16') for frame in movie_data]  # PIL images from np.ndarray
    #
    # # exec(f'import pyuda; client = pyuda.Client(); date_time = client.get_shot_date_time({pulse})')
    # # TODO: Remap header field names
    # # header_dict_subset = {key: value for key, value in header_dict.items()
    # #                       if key not in ('t_before_pulse', 'frame_times', 'n_frames', 'image_shape', 'detector_window')}
    # name_conventions = dict(pulse='shot', t_before_pulse='trigger',)

    # header_dict_subset = filter_kwargs(header_dict, funcs=(IpxHeader,), kwarg_aliases=name_conventions)

    # mastmovie assumes images are up-scaled to use full 16 bit dynamic range for displaying with PIL. Therefore
    # mastmovie downscales images before writing - compensate before write to reverse effect
    bit_depth_factor = 2**(16-depth)

    nuc_frame = copy(movie_data[1]) if (n_frames > 1) else copy(movie_data[0])
    if not apply_nuc:
        nuc_frame *= 0
        frames_ndarray = [frame for frame in movie_data*bit_depth_factor]  # for plotting with matplotlib
        frames = [Image.fromarray(frame, mode='I;16') for frame in frames_ndarray]  #
    else:
        logger.warning(f'Applying nuc subtraction to new IPX file: {ipx_path_fn}')
        frames_ndarray = [frame - nuc_frame for frame in movie_data*bit_depth_factor]  # for plotting with matplotlib
        frames = [Image.fromarray(frame - nuc_frame, mode='I;16') for frame in frames_ndarray]  #

    # frames = [Image.fromarray(np.uint8(cm.Greys(frame-nuc_frame))*255, mode='I;16') for frame in movie_data]  #
    # PIL images
    # frames = [frame.convert('I;16') for frame in frames]

    # fill in some header fields
    header = IpxHeader(**header_dict_subset)

    sensor = IpxSensor(
        type=SensorType.MONO, **sensor_dict
    )

    path_fn_ipx = Path(str(path_fn_ipx).format(**header_dict)).expanduser()

    if create_path:
        from fire.interfaces import io_basic
        io_basic.mkdir(path_fn_ipx)

    with write_ipx_file(
            path_fn_ipx, header, sensor, version=ipx_version,
            encoding=ImageEncoding.JPEG2K,
    ) as ipx:
        # write out the frames
        for i, (time, frame) in enumerate(zip(times, frames)):
            ipx.write_frame(time, frame)

    message = f'Wrote ipx file: "{path_fn_ipx}"'
    logger.debug(message)
    if verbose:
        print(message)

    return frames

def download_ipx_via_http_request(camera, pulse, path_out='~/data/movies/{machine}/{pulse}/',
                                  fn_out='{diag_tag_raw}0{pulse}.ipx', verbose=True):
    import requests
    from fire.interfaces.io_basic import mkdir

    machine = 'mast_u' if (pulse > 40000) else 'mast'

    path_out = Path(path_out.format(camera=camera, pulse=pulse, machine=machine)).expanduser()
    fn_out = fn_out.format(camera=camera, pulse=pulse, machine=machine)
    path_fn = path_out / fn_out
    mkdir(path_out, depth=2)

    url = f'http://video-replay-dev.mastu.apps.l/0{pulse}/{diag_tag_raw}/raw'
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

# read_movie_meta = read_movie_meta_with_pyipx
read_movie_meta = read_movie_meta_with_mastmovie
# read_movie_data = read_movie_data_with_pyipx
read_movie_data = read_movie_data_with_mastmovie

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # ipx_path = Path('../../../tests/test_data/mast/').resolve()
    # ipx_fn = 'rir030378.ipx'
    # ipx_path = Path('/home/tfarley/data/movies/mast_u/43651/rit/')
    ipx_path = Path('/home/tfarley/data/movies/mast_u/rit_ipx_files/')
    ipx_fn = 'rit044628.ipx'
    # ipx_fn = 'rit043651.ipx'

    ipx_path_fn = ipx_path / ipx_fn
    # ipx_path_fn = get_freia_ipx_path(29125, 'rit')

    meta_data = read_movie_meta_with_mastmovie(ipx_path / ipx_fn)
    print(meta_data)
    # frame_nos, frame_times, frame_data = read_movie_data_mastmovie(ipx_path / ipx_fn)
    # print(frame_times)

    meta_data = read_movie_meta(ipx_path_fn)
    frame_nos, frame_times, frame_data = read_movie_data(ipx_path_fn)
    print(meta_data)

    plt.imshow(frame_data[100], cmap='gray', interpolation='none')
    plt.show()
    pass