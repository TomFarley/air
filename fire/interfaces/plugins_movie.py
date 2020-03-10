# -*- coding: future_fstrings -*-
#!/usr/bin/env python

"""


Created: 
"""
import inspect
import logging
from typing import Union, Sequence, Tuple, Optional, Any, Dict

import numpy as np
from fire import fire_paths
from fire.interfaces.interfaces import PathList
from fire.utils import locate_files

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def read_movie_meta_data(pulse: Union[int, str], camera: str, machine: str, movie_plugins: dict,
                         movie_paths: Optional[PathList]=None, movie_fns: Optional[Sequence[str]]=None) \
                         -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Read movie header meta data

    Args:
        pulse           : Shot/pulse number or string name for synthetic movie data
        camera          : Camera diagnostic id/tag string
        machine         : Tokamak that the data originates from
        movie_plugins   : Dict of plugin functions for reading movie data
        movie_paths     : Search directories containing movie files
        movie_fns       : Movie filename format strings

    Returns: (meta data dictionary, data origin dictionary)

    """
    movie_meta_required_fields = ['n_frames', 'frame_range', 't_range', 'frame_shape', 'fps', 'lens', 'exposure',
                                  'bit_depth']
    plugin_key = 'meta'
    meta_data, origin = try_movie_plugins(plugin_key, pulse, camera, machine, movie_plugins,
                                          movie_paths=movie_paths, movie_fns=movie_fns)
    missing_fields = []
    for field in movie_meta_required_fields:
        if field not in meta_data:
            missing_fields.append(field)
    if len(missing_fields) > 0:
        raise ValueError(f'Movie plugin "{origin}" has not returned the folloing required meta data fields:\n'
                         f'{missing_fields}')
    return meta_data, origin

def read_movie_data(pulse: Union[int, str], camera: str, machine: str, movie_plugins: dict,
                    movie_paths: Optional[PathList]=None, movie_fns: Optional[Sequence[str]]=None, verbose: bool=True) \
                    -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, str]]:
    """Read movie frame data

    Args:
        pulse           : Shot/pulse number or string name for synthetic movie data
        camera          : Camera diagnostic id/tag string
        machine         : Tokamak that the data originates from
        movie_plugins   : Dict of plugin functions for reading movie data
        movie_paths     : Search directories containing movie files
        movie_fns       : Movie filename format strings

    Returns: frame_nos, frame_times, frame_data, origin
    """
    # TODO: Handle passing arguments for sub range of movie frames
    plugin_key = 'data'
    movie_data, origin = try_movie_plugins(plugin_key, pulse, camera, machine, movie_plugins,
                                           movie_paths=movie_paths, movie_fns=movie_fns)
    frame_nos, frame_times, frame_data = movie_data
    if verbose:
        logger.info(f'Read {len(frame_nos)} frames for camera "{camera}", pulse "{pulse}" from {str(origin)[1:-1]}')
    return frame_nos, frame_times, frame_data, origin

def try_movie_plugins(plugin_key, pulse, camera, machine, movie_plugins, movie_paths=None, movie_fns=None):
    """Iterate through calling supplied movie plugin functions in order supplied, until successful.

    Args:
        plugin_key      : Key for movie plugin function e.g. 'data'/'meta'
        pulse           : Pulse argument for movie plugin
        camera          : Camera argument for movie plugin
        machine         : Machine/tokamak argument for movie plugin
        movie_plugins   : Dict of movie plugin functions keyed: movie_plugins[<plugin_name>][<plugin_func>]
        movie_paths     : (Optional) Paths in which to look for movie files for file based plugins
        movie_fns       : (Optional) Filename patterns for movie files for file based plugins

    Returns: Tuple of data returned by plugin and a dictionary specifying the origin of the data (i.e. plugin,
             movie file etc.

    """
    kwargs = {'machine': machine, 'camera': camera, 'pulse': pulse, 'pulse_prefix': str(pulse)[0:2],
              'fire_path': str(fire_paths['root'])}
    data, origin = None, None
    for plugin_name, plugin_funcs in movie_plugins.items():
        read_movie_func = plugin_funcs[plugin_key]
        signature = inspect.signature(read_movie_func).parameters.keys()
        if ('path_fn' in signature):
            if (movie_paths is None) or (movie_fns is None):
                logger.warning(f'Skipping {plugin_name} movie plugin as movie path info not supplied')
                continue
            path_fns = locate_files(movie_paths, movie_fns, path_kws=kwargs, fn_kws=kwargs)
            if len(path_fns) == 0:
                continue
            for path_fn in path_fns:
                # Try reading each of the located possible movie files
                kwargs['path_fn'] = path_fn
                kws = {k: v for k, v in kwargs.items() if k in signature}
                try:
                    data = read_movie_func(**kws)
                    origin = {'plugin': plugin_name, 'path_fn': path_fn}
                    break
                except Exception as e:
                    continue
        else:
            try:
                data = read_movie_func(**kwargs)
                origin = {'plugin': plugin_name}
            except Exception as e:
                continue
        if data is not None:
            break
    if data is None:
        raise IOError(f'Failed to read movie data with movie plugins {list(movie_plugins.keys())} for args: {kwargs} ')
    return data, origin

if __name__ == '__main__':
    pass
