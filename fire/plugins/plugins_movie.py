# -*- coding: future_fstrings -*-
#!/usr/bin/env python

"""


Created:
"""
import inspect
import logging
from typing import Union, Sequence, Iterable, Tuple, Optional, Any, Dict
from copy import copy
from collections import namedtuple
from pathlib import Path

import numpy as np
from fire import fire_paths
from fire.interfaces.interfaces import PathList
from fire.misc.utils import dirs_exist, locate_files
from fire.plugins import plugins

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

class MovieReader:
    # TODO: Read plugin module attributes definition from file?
    # movie_plugin_definition_file = 'movie_plugin_definition.json'
    _plugin_paths = ["{fire_path}/plugins/movie_plugins/"]
    plugin_attributes = {
        "required": {
            "plugin_name": "movie_plugin_name",
            "meta": "read_movie_meta",
            "data": "read_movie_data",
            "info": "plugin_info"
        },
        "optional": {
        }
    }
    movie_paths = ["/net/fuslsc/data/MAST_Data/{pulse}/LATEST/",  # Directory path used by UDA
                    "~/data/movies/{machine}/{pulse}/{camera}/",
                    "/net/fuslsc.mast.l/data/MAST_IMAGES/0{pulse_prefix}/{pulse}/",
                    "/net/fuslsa/data/MAST_IMAGES/0{pulse_prefix}/{pulse}/",
                    "{fire_path}/../tests/test_data/{machine}/"]
    movie_fns = ["{camera}0{pulse}.ipx",
                 "{camera}_{pulse}.npz"]

    def __init__(self, movie_plugin_paths: Optional[PathList]=None, plugin_filter: Optional[Sequence[str]]=None,
                 plugin_precedence: Optional[Sequence[str]]=None,
                 movie_paths: Optional[PathList]=None, movie_fns: Optional[Sequence[str]]=None,
                 movie_plugin_definition_file: Optional[Union[str,Path]]=None):
        self.plugin_paths = movie_plugin_paths
        self.plugin_filter = plugin_filter
        self._plugin_precedence = plugin_precedence
        if movie_paths is not None:
            self.movie_paths = movie_paths
        if movie_fns is not None:
            self.movie_fns = movie_fns
        if movie_plugin_definition_file is not None:
            self.movie_plugin_definition_file = movie_plugin_definition_file
            raise NotImplementedError

        plugin_dicts, plugin_info = plugins.get_compatible_plugins(self.plugin_paths,
                               self.plugin_attributes['required'],
                               attributes_optional=self.plugin_attributes['optional'],
                               plugin_filter=plugin_filter, plugin_type='movie')
        self._plugin_dicts = plugin_dicts
        self._plugin_info = plugin_info
        self._plugins = {}
        self._active_plugin = None  # Plugin used for last successful read

        self._instanciate_plugins(plugin_dicts, plugin_info)

    def __repr__(self):
        plugins = self._plugins
        out = f'<MovieReader with {len(plugins)} plugins: {", ".join(plugins.keys())}>'
        return out

    def _instanciate_plugins(self, plugin_dicts, plugin_info=None):
        self._plugin_dicts = plugin_dicts
        self._plugin_info = plugin_info
        self._plugins = {}
        for name, plugin_dict in self._plugin_dicts.items():
            info = plugin_info[name] if (plugin_info is not None) and (name in plugin_info) else None

            self._plugins[name] = MoviePlugin.build_from_plugin_dict(name, plugin_dict, plugin_info=info)
        if len(self._plugins) == 0:
            raise IOError(f'Failed to load any movie reader plugins')

    def read_movie_meta_data(self, pulse: Union[int, str], camera: str, machine: str,
                         check_output: bool=True, substitute_unknown_values: bool=False) -> Tuple[Dict[str, Any],
                                                                                               Dict[str, str]]:
        for name, plugin in self.plugins.items():
            try:
                meta_data, origin = plugin.read_movie_meta_data(pulse=pulse, camera=camera, machine=machine,
                                                  movie_paths=self.movie_paths,
                                        movie_fns=self.movie_fns, check_output=check_output,
                                        substitute_unknown_values=substitute_unknown_values)
            except IOError as e:
                continue
            except Exception as e:
                raise e
            else:
                if meta_data is not None:
                    self._active_plugin = name
                    break
        else:
            raise IOError(f'Failed to read movie')
        return meta_data, origin

    def read_movie_data(self, pulse: Union[int, str], camera: str, machine: str,
                        n_start: Optional[int] = None, n_end: Optional[int] = None, stride: Optional[int] = 1,
                        frame_numbers: Optional[Union[Iterable, int]] = None,
                        transforms: Optional[Iterable[str]] = ()
                         # check_output: bool=True,
                        ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        for name, plugin in self.plugins.items():
            try:
                movie_data, origin = plugin.read_movie_data(pulse=pulse, camera=camera, machine=machine,
                                                    n_start=n_start, n_end=n_end, stride=stride,
                                                    frame_numbers=frame_numbers,
                                                    movie_paths=self.movie_paths, movie_fns=self.movie_fns,
                                                    transforms=transforms
                                                           # check_output=check_output,
                                                    )
            except IOError as e:
                continue
            except Exception as e:
                raise e
            else:
                if movie_data is not None:
                    self._active_plugin = name
                    break
        else:
            raise IOError(f'Failed to read movie')
        return movie_data, origin

    @property
    def plugins(self):
        """Return plugins dict ordered in active and precedence order"""
        keys = []
        if self._active_plugin is not None:
            keys.append(self._active_plugin)
        if self._plugin_precedence is not None:
            for key in self._plugin_precedence:
                if key not in keys:
                    keys.append(key)
        for key in self._plugins.keys():
            if key not in keys:
                keys.append(key)
        plugins = {key: self._plugins[key] for key in keys}
        return plugins

    @property
    def plugin_paths(self):
        return self._plugin_paths

    @plugin_paths.setter
    def plugin_paths(self, plugin_paths):
        # NOTE: Functionality duplicated in plugins.py?
        if plugin_paths is not None:
            self._plugin_paths = [Path(str(p).format(fire_path=fire_paths['root'])) for p in plugin_paths]

class MoviePlugin:
    module = None

    def __init__(self, name, methods, info=None):
        self.name = name
        self._methods = methods
        self._info = info
        pass

    def __repr__(self):
        out = f'<MoviePlugin:{self.name}, {self._info}>'
        return out

    @classmethod
    def build_from_plugin_dict(cls, name, plugin_dict, plugin_info=None):
        plugin = cls(name, plugin_dict, plugin_info)
        return plugin

    def read_movie_meta_data(self, pulse: Union[int, str], camera: str, machine: str,
                         movie_paths: Optional[PathList]=None, movie_fns: Optional[Sequence[str]]=None,
                         check_output: bool=True, substitute_unknown_values: bool=False) -> Tuple[Dict[str, Any],
                                                                                               Dict[str, str]]:
        """Read movie header meta data

        Args:
            pulse           : Shot/pulse number or string name for synthetic movie data
            camera          : Camera diagnostic id/tag string
            machine         : Tokamak that the data originates from
            movie_paths     : Search directories containing movie files
            movie_fns       : Movie filename format strings
            check_output    : Whether to run checks on format of outputed meta data

        Returns: (meta data dictionary, data origin dictionary)

        """
        movie_plugins = {self.name: self._methods}

        meta_data, origin = read_movie_meta_data(pulse, camera, machine, movie_plugins,
                                                    movie_paths=movie_paths, movie_fns=movie_fns,
                                                    check_output=check_output,
                                                    substitute_unknown_values=substitute_unknown_values)
        return meta_data, origin

    def read_movie_data(self, pulse: Union[int, str], camera: str, machine: str,
                    movie_paths: Optional[PathList]=None, movie_fns: Optional[Sequence[str]]=None,
                        n_start: Optional[int] = None, n_end: Optional[int] = None, stride: Optional[int] = 1,
                        frame_numbers: Optional[Union[Iterable, int]] = None,
                        transforms: Optional[Iterable[str]] = (),
                        verbose: bool=True
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, str]]:
        """Read movie frame data

        Args:
            pulse           : Shot/pulse number or string name for synthetic movie data
            camera          : Camera diagnostic id/tag string
            machine         : Tokamak that the data originates from
            movie_plugins   : Dict of plugin functions for reading movie data
            movie_paths     : Search directories containing movie files
            movie_fns       : Movie filename format strings
            verbose         : Print to console

        Returns: frame_nos, frame_times, frame_data, origin
        """
        movie_plugins = {self.name: self._methods}

        movie_data, origin = read_movie_data(pulse, camera, machine, movie_plugins,
                                             n_start=n_start, n_end=n_end, stride=stride,
                                             frame_numbers=frame_numbers,
                                             movie_paths=movie_paths, movie_fns=movie_fns,
                                             transforms=transforms, verbose=verbose)
        return movie_data, origin


def read_movie_meta_data(pulse: Union[int, str], camera: str, machine: str, movie_plugins: dict,
                         movie_paths: Optional[PathList]=None, movie_fns: Optional[Sequence[str]]=None,
                         check_output: bool=True, substitute_unknown_values: bool=False) -> Tuple[Dict[str, Any],
                                                                                               Dict[str, str]]:
    """Read movie header meta data

    Args:
        pulse           : Shot/pulse number or string name for synthetic movie data
        camera          : Camera diagnostic id/tag string
        machine         : Tokamak that the data originates from
        movie_plugins   : Dict of plugin functions for reading movie data
        movie_paths     : Search directories containing movie files
        movie_fns       : Movie filename format strings
        check_output    : Whether to run checks on format of outputed meta data

    Returns: (meta data dictionary, data origin dictionary)

    """
    # TODO: Make all plugins return consistent format for exposure - currently have either str or float?
    movie_meta_required_fields = ['n_frames', 'frame_range', 't_range', 'image_shape', 'fps', 'lens', 'exposure',
                                  'bit_depth']
    plugin_key = 'meta'
    meta_data, origin = try_movie_plugins_dicts(plugin_key, pulse, camera, machine, movie_plugins,
                                                movie_paths=movie_paths, movie_fns=movie_fns)

    rename_meta_data_fields(meta_data)
    check_movie_meta_data(meta_data, required_fields=movie_meta_required_fields, check_bad_values=True,
                          substitute_unknown_values=substitute_unknown_values)
    meta_data = reformat_movie_meta_data(meta_data)
    meta_data = add_alternative_meta_data_representations(meta_data)

    return meta_data, origin

def read_movie_data(pulse: Union[int, str], camera: str, machine: str, movie_plugins: dict,
                    n_start:Optional[int]=None, n_end:Optional[int]=None, stride:Optional[int]=1,
                    frame_numbers: Optional[Union[Iterable, int]]=None,
                    movie_paths: Optional[PathList]=None, movie_fns: Optional[Sequence[str]]=None,
                    transforms: Optional[Iterable[str]] = (), verbose: bool=True) \
                    -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, str]]:
    """Read movie frame data

    Args:
        pulse           : Shot/pulse number or string name for synthetic movie data
        camera          : Camera diagnostic id/tag string
        machine         : Tokamak that the data originates from
        movie_plugins   : Dict of plugin functions for reading movie data
        movie_paths     : Search directories containing movie files
        movie_fns       : Movie filename format strings
        verbose         : Print to console

    Returns: namedtuple: (frame_nos, frame_times, frame_data), origin
    """
    # TODO: Handle passing arguments for sub range of movie frames
    plugin_key = 'data'
    movie_data, origin = try_movie_plugins_dicts(plugin_key, pulse, camera, machine, movie_plugins,
                                                 n_start=n_start, n_end=n_end, stride=stride,
                                                 frame_numbers=frame_numbers,
                                                 movie_paths=movie_paths, movie_fns=movie_fns,
                                                 transforms=transforms)

    # Convert tuples to named tuples
    movie_data_tup = namedtuple('movie_data', ('frame_nos', 'frame_times', 'frame_data'))
    movie_data_and_origin_tup = namedtuple('movie_data_and_origin', ('movie_data', 'origin'))
    movie_data_and_origin = movie_data_and_origin_tup(movie_data_tup(*movie_data), origin)

    if verbose:
        logger.info(f'Read {len(movie_data[0])} frames for camera "{camera}", pulse "{pulse}" from {str(origin)[1:-1]}')
    return movie_data_and_origin

def try_movie_plugins_dicts(plugin_key, pulse, camera, machine, movie_plugins,
                            n_start:Optional[int]=None, n_end:Optional[int]=None, stride:Optional[int]=1,
                            frame_numbers: Optional[Union[Iterable, int]]=None,
                            movie_paths:Optional[PathList]=None, movie_fns=None,
                            transforms: Optional[Iterable[str]]=(),
                            func_kwargs=None):
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
    kwargs = {'machine': machine, 'camera': camera, 'pulse': pulse,
              'n_start': n_start, 'n_end': n_end, 'stride': stride, 'frame_numbers': frame_numbers,
              'movie_paths': movie_paths, 'movie_fns': movie_fns,
              'transforms': transforms,
              'pulse_prefix': str(pulse)[0:2],
              'fire_path': str(fire_paths['root'])}
    data, origin = None, None
    for plugin_name, plugin_funcs in movie_plugins.items():
        movie_func = plugin_funcs[plugin_key]
        status, kws, origin_options = get_movie_plugin_args(movie_func, kwargs, func_kwargs,
                                                           movie_paths=movie_paths, movie_fns=movie_fns)
        if status != 'ok':
            logger.debug(status)
        else:
            origin = {}
            if len(origin_options) > 0:
                # Look over multiple possible movie locations/origins
                for kwarg_name, options in origin_options.items():
                    for option in options:
                        origin[kwarg_name] = option
                        kws.update(origin)
                        try:
                            data = movie_func(**kws)
                            origin = {'plugin': plugin_name, **origin}
                            if kwarg_name == 'path_fn':
                                origin['path'] = option.parent
                                origin['fn'] = option.name
                            break
                        except TypeError as e:
                            raise e
                        except Exception as e:
                            logger.debug(f'Failed to call "{movie_func.__name__}" from "{movie_func.__module__}" '
                                         f'with args {kws}: {e}')
                            continue
                    if data is not None:
                        break
            else:
                # Only one set of arguments to try eg pyUDA
                try:
                    data = movie_func(**kws)
                    origin = {'plugin': plugin_name}
                except Exception as e:
                    logger.debug(f'Failed to call "{movie_func.__name__}" from "{movie_func.__module__}" '
                                 f'with args {kws}')
                    continue

        if data is not None:
            # Use first successful path options - multiple valid movie paths may exist for same movie
            break
    if data is None:
        raise IOError(f'Failed to read movie data with movie plugins {list(movie_plugins.keys())} from paths:\n '
                      f'{movie_paths} \n'
                      f'with filename formats: \n'
                      f'{movie_fns} \n'
                      f'for plugin args: \n'
                      f'{kwargs}')
    origin_str = ", ".join([f'{k}="{str(v)}"' for k, v in origin.items() if k not in ['plugin', 'path_fn']])
    logger.info(f'Read movie "{plugin_key}" for ({machine}, {camera}, {pulse}) using plugin "{origin["plugin"]}": '
                f'{origin_str}')

    return data, origin

def get_movie_plugin_args(read_movie_func, kwargs_generic=None, kwargs_specific=None, movie_paths=None, movie_fns=None):
    signature = inspect.signature(read_movie_func).parameters.keys()
    kwargs = copy(kwargs_generic)
    # Default fail values
    kws = {}
    origin_options = {}
    if ('path_fn' in signature):
        # Direct path to movie file eg IPX file
        if (movie_paths is None) or (movie_fns is None):
            # No paths or filenames supplied
            status = f'Movie path info not supplied'
            return status, kws, origin_options

        path_fns = locate_files(movie_paths, movie_fns, path_kws=kwargs, fn_kws=kwargs)
        if len(path_fns) == 0:
            status = f'No movie files located in: {movie_paths}\nwith parameters: {kwargs}'
            return status, kws, origin_options
        origin_options['path_fn'] = path_fns

    if ('path' in signature):
        # Path containing multiple files eg. imstack movie
        paths_exist, paths_raw_exist, paths_raw_not_exist = dirs_exist(movie_paths, path_kws=kwargs)
        if len(paths_exist) == 0:
            status = f'No movie file directories located in: {movie_paths}'
            return status, kws, origin_options
        origin_options['path'] = paths_exist

    kws = {k: v for k, v in kwargs.items() if k in signature}
    status = 'ok'
    return status, kws, origin_options

def rename_meta_data_fields(meta_data):
    name_mapping = dict(frame_shape='image_shape')

    for old_name in name_mapping.keys():
        if old_name in meta_data:
            new_name = name_mapping[old_name]
            meta_data[new_name] = meta_data.pop(old_name)
            logger.debug('Renamed meta data field "%s" to "%s"', old_name, new_name)

def reformat_movie_meta_data(meta_data):
    """Convert meta data to consistent FIRE conventions - in particular SI units

    Args:
        meta_data: Dict of meta data to reformat

    Returns: Dict of reformatted meta data

    """

    # Convert lens field from string eg "13mm" to float focal length in meters
    key = 'lens'
    value = meta_data.get(key, None)
    if isinstance(value, str):
        try:
            # TODO: Use regex to handle while space before unit
            if 'mm' in value:
                value_new = float(value.split('mm')[0])*1e-3  # Convert mm to meters
            elif 'cm' in value:
                value_new = float(value.split('cm')[0])*1e-2  # Convert cm to meters
            else:
                value_new = float(value)  # assume already in meters
                logger.debug(f'Assuming string value of lens focal length already in meters')
        except ValueError as e:
            logger.warning(f'Failed to convert string value for lens "{value}" to float')
        else:
            meta_data[key] = value_new
            logger.debug(f'Converted meta data "{key}" to float in meters: "{value}" -> {value_new:0.4g}')

    # Convert exposure time field from micro seconds to seconds eg 232 to 0.000232
    key = 'exposure'
    value = meta_data.get(key, None)
    if isinstance(value, (int, float)) and (value > 1):
        value_new = value*1e-6  # Convert mm to meters
        meta_data[key] = value_new
        logger.debug(f'Converted meta data "{key}" from us to s: "{value}" -> {value_new:0.4g}')

    return meta_data

def add_alternative_meta_data_representations(meta_data):
    """Add additional entries to meta data dict that are eg useful in format strings.

    The input meta data dict should be passed through reformat_movie_meta_data() first.

    Args:
        meta_data: Movie meta data dict

    Returns: Movie meta data dict with additional entries

    """
    key = 'lens'
    value = meta_data.get(key, None)
    if value:
        try:
            meta_data['lens_in_cm'] = int(value*1e2)
        except TypeError as e:
            logger.exception(f'Failed to convert lens focal length value "{value}" to cm')
            raise e

    key = 'exposure'
    value = meta_data.get(key, None)
    if value:
        meta_data['exposure_in_us'] = int(value*1e6)

    return meta_data

def check_movie_meta_data(meta_data, required_fields=None, check_bad_values=True, substitute_unknown_values=False):
    """Check movie meta data dictionary contains required values for FIRE analysis and substitute/update values
    inplace where appropriate

    Args:
        meta_data                   : Movie meta data dictionary
        required_fields             : List of field names that must be present
        check_bad_values            : Bool for whether to check for bad (nan/'Unknown') values
        substitute_unknown_values   : Whether to substitute default values for missing data (intended for debugging)

    Returns:

    """
    if substitute_unknown_values is True:
        substitute_unknown_values = {'lens': 25e-2, 'exposure': 25e-6}
    if required_fields is not None:
        missing_fields = []
        for field in required_fields:
            if field not in meta_data:
                missing_fields.append(field)
        if len(missing_fields) > 0:
            raise ValueError(f'Movie plugin "{origin}" has not returned the folloing required meta data fields:\n'
                             f'{missing_fields}')
    if check_bad_values:
        requried_keys = ['lens', 'exposure', 'detector_window']
        bad_values = ('Unknown', 'unknown', np.nan)
        for key in requried_keys:
            value = meta_data[key]
            if ((value is None) or
                    (isinstance(value, str)) and (value in bad_values) or
                    ((not isinstance(value, (str))) and np.any(np.isnan(value)))):
                if substitute_unknown_values is not False:
                    if key in substitute_unknown_values:
                        meta_data[key] = substitute_unknown_values[key]
                        logger.warning(f'Substituted "{key}" default value of {meta_data[key]} for bad value "{value}"')
                else:
                    logger.warning(f'Bad movie meta data value for "{key}": {value}')

    detector_window = meta_data['detector_window']
    if np.any(np.isnan(meta_data['detector_window'])):
        # (Left,Top,Width,Height)
        image_shape = np.array(meta_data['image_shape'])
        # TODO: Check if (some?) detector_window values should be incremented by 1?
        if np.isnan(detector_window[0]) and (image_shape[1] >= 256):
            detector_window[0] = 0  # +1
            logger.warning(f'Assuming missing "left" value in detector_window[0] to be 0: {detector_window}')
        if np.isnan(detector_window[1]) and (image_shape[0] >= 256):
            detector_window[1] = 0  # +1
            logger.warning(f'Assuming missing "top" value in detector_window[1] to be 0: {detector_window}')

    if np.any(np.isnan(meta_data['detector_window'])):
        raise ValueError(f'Movie data appears to be sub-windowed without detector_window meta data')
    else:
        meta_data['detector_window'] = meta_data['detector_window'].astype(int)

    if np.all(detector_window[0:2] > 0):
        detector_window[0:2] -= 1
        meta_data['detector_window'] = detector_window
        # TODO: Compare to image_shape to check validity?
        logger.warning(f'Subtracted 1 from detector window corner coordinates (need to check validity): {detector_window}')
    pass


if __name__ == '__main__':
    pass
