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
import xarray as xr

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

MovieData = namedtuple('movie_plugin_frame_data', ['frame_numbers', 'frame_times', 'frame_data'])

movie_plugin_name = 'ipx'
plugin_info = {'description': 'This plugin reads IPX1/2 format MAST movie files'}

def get_freia_ipx_path(pulse, camera):
    """Return path to ipx file on UKAEA freia cluster

    :param pulse: Shot/pulse number or string name for synthetic movie data
    :param camera: Name of camera to analyse (unique name of camera or diagnostic code)
    :return: Path to ipx files
    """
    pulse = str(pulse)
    # MAST_IMAGES = '/net/fuslsa/data/MAST_IMAGES/'
    MAST_IMAGES = '/net/fuslsc.mast.l/data/MAST_IMAGES/'
    ipx_path_fn = MAST_IMAGES + f"0{pulse[0:2]}/{pulse}/{camera}0{pulse}.ipx"
    return ipx_path_fn

def read_movie_meta(path_fn: Union[str, Path], transforms: Iterable[str]=()) -> dict:
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
    last_frame = ipx_header['n_frames'] - 1

    # Last frame doesn't always load, so work backwards to last successfully loaded frame
    ret = False
    while not ret:
        # Read last frame
        vid.set_frame_number(last_frame)
        ret, frame_end, frame_header_end = vid.read(transforms=transforms)
        if not ret:
            # File closes when it fails to load a frame, so re-open
            vid = ipxReader(filename=path_fn)
            last_frame -= 1
    vid.release()

    # Collect summary of ipx file meta data
    # file_header['ipx_version'] = vid.ipx_type
    movie_meta = {'movie_format': '.ipx'}
    movie_meta['n_frames'] = ipx_header['n_frames']
    movie_meta['frame_range'] = np.array([0, last_frame])
    movie_meta['t_range'] = np.array([float(frame_header0['time_stamp']), float(frame_header_end['time_stamp'])])
    movie_meta['t_before_pulse'] = np.abs(ipx_header.get('trigger', movie_meta['t_range'][0]))
    movie_meta['image_shape'] = np.array(frame0.shape)
    movie_meta['fps'] = (last_frame) / np.ptp(movie_meta['t_range'])
    movie_meta['exposure'] = ipx_header['exposure']
    movie_meta['bit_depth'] = ipx_header['depth']
    movie_meta['lens'] = ipx_header['lens'] if 'lens' in ipx_header else 'Unknown'
    # TODO: Add filter name?

    # TODO: Move derived fields to common function for all movie plugins: image_shape, fps, t_range
    # TODO: Check ipx field 'top' follows image/calcam conventions
    movie_meta['detector_window'] = get_detector_window_from_ipx_header(ipx_header, plugin='ipx', fn=path_fn)

    movie_meta['ipx_header'] = ipx_header
    return movie_meta

def get_detector_window_from_ipx_header(ipx_header, plugin=None, fn=None):
    # TODO: Move to generic movie plugin functions module

    left, top, width, height, right, bottom = np.array([ipx_header[key] if key in ipx_header else np.nan for key in
                                                         ('left', 'top', 'width', 'height', 'right', 'bottom')])
    detector_window = np.array([left, top, width, height])

    if np.any(np.isnan(detector_window)):
        logger.warning(f'IPX file missing meta data for detector sub-window: {detector_window} ({plugin}, {fn})')

    try:
        width_calc = right - left
        width = ipx_header['width']
    except KeyError as e:
        pass
    else:
        if width_calc != width:
            logger.warning(f'Detector window width calculated from right-left = {right}-{left} = {width_calc} '
                           f'!= {width} = width')

    try:
        height_calc = top - bottom
        height = ipx_header['height']
    except KeyError as e:
        pass
    else:
        if height_calc != height:
            logger.warning(f'Detector window height calculated from top-bottom = {top}-{bottom} = {height_calc} '
                           f'!= {height} = height')

    if not np.any(np.isnan(detector_window)):
        detector_window = detector_window.astype(int)

    return detector_window

def convert_ipx_header_to_uda_conventions(header: dict) -> dict:
    """

    :param header: Ipx header dict with each parameter a separate scalar value
    :return: Reformatted header dict
    """
    # TODO: Make generic functions for use in all plugins: rename_dict_keys, convert_types, modify_values
    header = copy(header)  # Prevent changes to pyIpx movieReader attribute still used for reading video
    # TODO: Update values due to changes in uda output conventions (now using underscores?)
    key_map = {'numFrames': 'n_frames', 'codec': 'codex', 'color': 'is_color', 'ID': 'ipx_version', 'hBin': 'hbin',
                'vBin': 'vbin', 'datetime': 'date_time', 'preExp': 'pre_exp', 'preexp': 'pre_exp',
               'ipx_version': 'file_format', 'orient': 'orientation'}
    missing = []
    for old_key, new_key in key_map.items():
        if new_key in header:
            pass  # Already has correct name
        else:
            try:
                header[new_key] = header.pop(old_key)
                logger.debug(f'Renamed ipx header parameter from "{old_key}" to "{new_key}".')
            except KeyError as e:
                missing.append(old_key)
    if len(missing) > 0:
        missing = {k: key_map[k] for k in missing}
        logger.warning(f'Could not rename {len(missing)} ipx header parameters as original keys missing: '
                       f'{missing}')

    missing_scalar_keys = []
    if 'gain' not in header:
        try:
            header['gain'] = [header.pop('gain_0'), header.pop('gain_1')]
        except KeyError as e:
            missing_scalar_keys.append('gain')
    if 'offset' not in header:
        try:
            header['offset'] = [header.pop('offset_0'), header.pop('offset_1')]
        except KeyError as e:
            missing_scalar_keys.append('offset')
    if len(missing_scalar_keys) > 0:
        logger.warning(f'Could not build missing ipx header parameters {missing_scalar_keys} as scalar params also '
                       f'missing')

    if 'size' in header:
        # Size field is left over from binary header size for IPX1? reader - not useful
        header.pop('size')

    # header['bottom'] = header.pop('top') - header.pop('height')  # TODO: check - not +
    # header['right'] = header.pop('left') + header.pop('width')
    return header

def read_movie_data(path_fn: Union[str, Path],
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
            # frames are read with 16 bit dynamic range, but values are 10 bit!
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
    from mastvideo import load_ipx_file, VideoDecoder

    if not Path(path_fn).is_file():
        raise FileNotFoundError(f'IPX file does not exist: {path_fn}')

    # open the file
    ipx = load_ipx_file(open(path_fn, mode='rb'))
    ipx.header.validate()

    # display information about the file
    # print(ipx.header)
    print(ipx.sensor)

    movie_meta = dict(movie_format='.ipx')

    header_fields = {'num_frames': 'n_frames',
                     'exposure': 'exposure',
                     'depth': 'bit_depth',
                     'lens': 'lens',
                     'board_temperature': 'board_temperature',
                     'bytes_per_decoded_frame': 'bytes_per_decoded_frame',
                     'camera': 'camera',
                     'count': 'count',
                     'date_time': 'date_time',
                     'filter': 'filter',
                     'frame_height': 'height',
                     'frame_width': 'width',
                     'index': 'index',
                     'orientation': 'orientation',
                     'pil_image_mode': 'pil_image_mode',
                     'pixels_per_frame': 'pixels_per_frame',
                     'pre_exposure': 'pre_exposure',
                     'sensor_temperature': 'sensor_temperature',
                     'shot': 'shot',
                     'strobe': 'strobe',
                     'trigger': 'trigger',
                     'view': 'view',
    }

    movie_meta.update({name: getattr(ipx.header, key) for key, name in header_fields.items()})

    sensor_fields = {
                     'binning_h': 'binning_h',
                     'binning_v': 'binning_v',
                     'count': 'count',
                     'gain': 'sensor_gain',
                     'index': 'index',
                     'offset': 'offset',
                     'taps': 'taps',
                     'type': 'sensor_type',
                     'window_bottom': 'bottom',
                     'window_left': 'left',
                     'window_right': 'right',
                     'window_top': 'top'
                      }
    # TODO: Rename meta data window fields to window_<> for all plugins/pass through reformat function
    movie_meta.update({name: getattr(ipx.sensor, key) for key, name in sensor_fields.items()})


    movie_meta['frame_range'] = np.array([0, movie_meta['n_frames']])
    movie_meta['t_range'] = np.array([ipx.frames[0].time, ipx.frames[-1].time])  # TODO: Refine
    movie_meta['image_shape'] = np.array([movie_meta['height'], movie_meta['width']])
    movie_meta['fps'] = movie_meta['n_frames'] / (movie_meta['t_range'][1] - movie_meta['t_range'][0])
    movie_meta['detector_window'] = get_detector_window_from_ipx_header(movie_meta, plugin='ipx', fn=path_fn)

    frame_times = np.array([frame.time for frame in ipx.frames])
    movie_meta['fps'] = 1 / np.median(np.diff(frame_times))

    # raise NotImplementedError
    return movie_meta


def read_movie_data_with_mastmovie(path_fn: Union[str, Path],
                                   n_start: Optional[int] = None, n_end: Optional[int] = None,
                                   stride: Optional[int] = 1,
                                   frame_numbers: Optional[Union[Iterable, int]] = None,
                                   transforms: Optional[Iterable[str]] = (),
                                   verbose: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from mastvideo import load_ipx_file, VideoDecoder

    if not Path(path_fn).is_file():
        raise FileNotFoundError(f'IPX file does not exist: {path_fn}')


    # open the file
    ipx = load_ipx_file(open(path_fn, mode='rb'))

    # video = VideoDecoder(ipx)  # convert to 8bit (interpolate to RGB, if valid for sensor) for conversion to mpeg video
    # frames = list(video.frames())

    n_frames = ipx.header.num_frames

    frame_data = np.zeros((n_frames, ipx.header.frame_height, ipx.header.frame_width))

    n_start = 0 if (n_start is None) else n_start
    n_end = n_frames-1 if (n_end is None) else n_end
    frame_numbers = np.arange(n_start, n_end+1, stride)

    frame_times = np.array([frame.time for frame in ipx.frames])

    # TODO: Deal with non n=0 first frame?
    # iterate over frames converting to numpy array
    i = 0
    for n, frame in enumerate(ipx.frames):
        if n in frame_numbers:
            image = ipx.decode_frame(frame.data)  # 'I;16'
            frame_data[i] = np.array(image)
            i += 1
    assert i == n_frames, f"i != n_frames: {i} != {n_frames}"

    message = f'Read ipx file with mastmovie: "{path_fn}"'
    logger.debug(message)
    if verbose:
        print(message)

    return frame_numbers, frame_times, frame_data

def write_ipx_with_mastmovie(path_fn_ipx: Union[Path, str], movie_data: np.ndarray, header_dict: dict,
                             apply_nuc=False, verbose: bool=True):
    from PIL import Image
    from mastvideo import write_ipx_file, IpxHeader, IpxSensor, SensorType, ImageEncoding
    from matplotlib import cm
    from fire.scripts.organise_ircam_raw_files import complete_meta_data_dict
    from fire.misc.utils import filter_kwargs

    movie_data = movie_data.astype(np.uint16)

    n_frames, height, width = tuple(movie_data.shape)
    image_shape = (height, width)

    pulse = header_dict.get('shot', header_dict.get('pulse'))
    camera = header_dict.get('camera', 'IRCAM_Velox81kL_0102')
    times = header_dict['frame_times']

    trigger = header_dict.get('t_before_pulse', header_dict.get('trigger',
                                                                header_dict.get('ipx_header', {}).get('trigger')))

    # TODO: Use pycpf when it working to get datetime eg '2013-06-11T14:27:21'
    header_dict_subset = dict(shot=pulse, date_time='<placeholder>', camera=camera,
                               view='HL04_A-tangential', lens='25 mm', trigger=-np.abs(trigger),
                               exposure=int(header_dict['exposure'] * 1e6), num_frames=n_frames, frame_width=width,
                               frame_height=height, depth=14,
                               # codec='jp2',
                              # file_format='IPX 01',
                               # offset=0.0, ccdtemp=-1, numFrames, filter, codex='JP2', ID='IPX 01'
        )
    if image_shape == (256, 320):
        # header_dict_subset['top'] = 0
        # header_dict_subset['bottom'] = 0
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

    nuc_frame = copy(movie_data[1])
    if not apply_nuc:
        nuc_frame *= 0
    frames_ndarray = [frame - nuc_frame for frame in movie_data]  # for plotting with matplotlib
    frames = [Image.fromarray(frame-nuc_frame, mode='I;16') for frame in frames_ndarray]  #
    # frames = [Image.fromarray(np.uint8(cm.Greys(frame-nuc_frame))*255, mode='I;16') for frame in movie_data]  #
    # PIL images
    # frames = [frame.convert('I;16') for frame in frames]

    # fill in some dummy fields
    header = IpxHeader(**header_dict_subset)

    sensor = IpxSensor(
        type=SensorType.MONO,
    )

    path_fn_ipx = Path(str(path_fn_ipx).format(**header_dict)).expanduser()

    with write_ipx_file(
            path_fn_ipx, header, sensor, version=1,
            encoding=ImageEncoding.JPEG2K,
    ) as ipx:
        # write out the frames
        for time, frame in zip(times, frames):
            ipx.write_frame(time, frame)

    message = f'Wrote ipx file: "{path_fn_ipx}"'
    logger.debug(message)
    if verbose:
        print(message)

    return frames

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
    import matplotlib.pyplot as plt
    # ipx_path = Path('../../../tests/test_data/mast/').resolve()
    # ipx_fn = 'rir030378.ipx'
    # ipx_path = Path('/home/tfarley/data/movies/mast_u/43651/rit/')
    # ipx_fn = 'rit043651.ipx'
    #
    # ipx_path_fn = ipx_path / ipx_fn

    ipx_path_fn = get_freia_ipx_path(29125, 'rit')

    # meta_data = read_movie_meta_mastmovie(ipx_path / ipx_fn)
    # print(meta_data)
    # frame_nos, frame_times, frame_data = read_movie_data_mastmovie(ipx_path / ipx_fn)
    # print(frame_times)

    meta_data = read_movie_meta(ipx_path_fn)
    frame_nos, frame_times, frame_data = read_movie_data(ipx_path_fn)
    print(meta_data)

    plt.imshow(frame_data[100], cmap='gray', interpolation='none')
    plt.show()
    pass