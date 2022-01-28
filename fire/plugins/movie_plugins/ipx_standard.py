#!/usr/bin/env python

"""


Created: 
"""

import logging
from copy import copy
from typing import Union, Iterable, Sequence, Tuple, Optional, Any, Dict
from pathlib import Path

import numpy as np

from fire.misc import utils

logger = logging.getLogger(__name__)
logger.propagate = False

UDA_IPX_HEADER_FIELDS = ('board_temp', 'camera', 'ccd_temp', 'codex', 'date_time', 'depth', 'exposure', 'file_format',
                         'filter', 'frame_times', 'gain', 'hbin', 'height', 'is_color', 'left', 'lens',
                         'n_frames', 'offset', 'orientation', 'pre_exp', 'shot', 'strobe', 'taps', 'top', 'trigger',
                         'vbin', 'view', 'width')

# UDA_IPX_HEADER_FIELDS = ('board_temp', 'camera', 'ccd_temp', 'date_time', 'depth', 'exposure', 'filter', 'frame_times',
#                          'gain', 'hbin', 'height', 'is_color', 'left', 'lens', 'n_frames', 'offset', 'pre_exp', 'shot',
#                          'taps', 'top', 'vbin', 'view', 'width')
# uda_ipx_header_fields = ('board_temp', 'camera', 'ccd_temp', 'datetime', 'depth', 'exposure', 'filter', 'frame_times',
#                          'gain', 'hbin', 'height', 'is_color', 'left', 'lens', 'n_frames', 'offset', 'preexp', 'shot',
#                          'taps', 'top', 'vbin', 'view', 'width')
# uda_ipx_header_fields += ('ID', 'size', 'codec', 'date_time', 'trigger', 'orient', 'color', 'hBin',
#                           'right', 'vBin', 'bottom', 'offset_0', 'offset_1', 'gain_0', 'gain_1', 'preExp', 'strobe')

def check_ipx_detector_window_meta_data(ipx_header, plugin=None, fn=None, modify_inplace=True,
                                        missing_value=np.nan, verbose=True):
    # TODO: Move to generic movie plugin functions module
    log = logger.warning if verbose else logger.debug
    problems = 0

    # TODO: apply abs to each value so don't have negative 'bottom' value etc?
    left, top, width, height, right, bottom = [np.abs(
                                               utils.str_to_number(ipx_header.get(key, missing_value), cast=int,
                                                        if_not_numeric='return_default', default_non_numeric=np.nan)
                                                     )
                                                      for key in ('left', 'top', 'width', 'height', 'right', 'bottom')]

    left_top = np.array([left, top])
    width_height = np.array([width, height])

    if np.any(np.isnan(width_height)):
        raise ValueError(f'Ipx header dict is missing width/height meta data: {width_height}')

    # From IPX1 documentation: left: offset=244, length=2, type=unsigned; Window position (leftmost=1); 0 – not defined.
    # Therefore subtract 1 to get zero indexed pixel coordinate
    if np.any(left_top == 0):
        log(f'{plugin} Detector window corner coords contain zeros which for ipx standard means "not defined": '
            f'(left, top = {left}, {top}) ({plugin}, {fn})')
        left_top[(left_top == 0)] = 1
        left, top = left_top.astype(int)
        log(f'Assumed window corner at origin (leftmost=1): {left_top}')
        if np.any(width_height < 256):
            raise ValueError(f'Detector appears to be subwindowed, so assuming corner at origin probably not valid')
        problems += 1

    if np.any(np.isnan(left_top)):
        log(f'{plugin} file missing meta data for detector sub-window origin: {left_top} ({plugin}, {fn})')
        left_top[np.isnan(left_top)] = 1
        left, top = left_top.astype(int)
        log(f'Set left, top nans to origin (leftmost=1): {left_top}')
        problems += 1

    if np.isnan(right) or (right == 0):
        right = int(left + width)
        logger.debug(f'Calculated sub-window "right" = left + width = {right}')
    else:
        width_calc = right - left
        if width_calc != width:
            log(f'Detector window width calculated from right-left = {right}-{left} = {width_calc} '
                           f'!= {width} = width')
            problems += 1

    if np.isnan(bottom) or (bottom == 0):
        bottom = int(top + height)
        logger.debug(f'Calculated sub-window "bottom" = top + height = {bottom}')
    else:
        height_calc = bottom - top
        if height_calc != height:
            log(f'Detector window height calculated from bottom-top = {bottom}-{top} = {height_calc} '
                           f'!= {height} = height')
            problems += 1

    if problems > 0:
        pass

    image_resolution = ipx_header.get('image_resolution', ipx_header.get('image_shape'))
    if image_resolution is not None:  # Not standard ipx field, but used in FIRE - equal to frame_data.shape[1:]
        if not np.all(image_resolution == np.array([height, width])):
            raise ValueError('Image_resolution field doesnt match other header meta data')

    if modify_inplace:
        for key, value in zip(('left', 'top', 'width', 'height', 'right', 'bottom'),
                              (left, top, width, height, right, bottom)):
            ipx_header[key] = int(value)

    return (left, top, width, height, right, bottom)


def get_detector_window_from_ipx_header(ipx_header):
    """Return tuple of ('left', 'top', 'width', 'height') with left and top starting at 0 (not 1 as in ipx standard).
    Note: Coordinates and widths should be according to 'Original' coords not 'Display' coords (Calcam conventions)"""
    # TODO: apply abs to each value so don't have negative 'bottom' value etc?
    left, top, width, height, right, bottom = np.array([np.abs(ipx_header[key]) if key in ipx_header else np.nan
                                                      for key in ('left', 'top', 'width', 'height', 'right', 'bottom')])

    # From IPX1 documentation: left: offset=244, length=2, type=unsigned; Window position (leftmost=1); 0 – not defined.
    # Therefore subtract 1 to get zero indexed pixel coordinate
    # Switch to Calcam conventions:
    #  Order: (Left,Top,Width,Height)
    #  Index: Start from 0 (as opposed to 1 for IPX index conventions)
    detector_window = np.array([left-1, top-1, width, height])

    if not np.any(np.isnan(detector_window)):
        detector_window = detector_window.astype(int)
    else:
        raise ValueError(f'Detector sub-window contains nans: {detector_window}')

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
        logger.debug(f'Could not rename {len(missing)} ipx header parameters as original keys missing: '
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

if __name__ == '__main__':
    pass