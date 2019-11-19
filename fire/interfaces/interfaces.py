#!/usr/bin/env python
"""
The `interfaces` module contains functions for interfacing with other codes and files.
"""

import os, sys, glob, logging, json, inspect, traceback
import importlib.util
from typing import Union, Sequence, Optional, Tuple, List, Dict, Any
from pathlib import Path

import numpy as np
import pandas as pd

from fire.utils import locate_file, locate_files, join_path_fn, make_iterable, PathList
from fire import fire_paths

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PathList = Sequence[Union[Path, str]]

def format_path(path: Union[str, Path] , **kwargs) -> Path:
    kwargs.update({'fire_path': fire_paths['root']})
    path = Path(str(path).format(**kwargs)).expanduser()
    return path

def identify_files(pulse, camera, machine, search_paths_inputs=None, fn_patterns_inputs=None,
                   paths_output=None, fn_pattern_output=None,
                   params=None):
    """Return dict of paths to input files needed for IR analysis

    :param pulse: Shot/pulse number or string name for synthetic movie data
    :param camera: Name of camera to analyse (unique name of camera or diagnostic code)
    :param machine: Tokamak that the data originates from
    :return: Dict of filepaths
    """
    if search_paths_inputs is None:
        search_paths_inputs = ["~/fire/input_files/{machine}/", "{fire_path}/input_files/{machine}/", "~/calcam/calibrations/"]
    if fn_patterns_inputs is None:
        # TODO: UPDATE
        fn_patterns_inputs = {"calcam_calibs": ["calcam_calibs-{machine}-{camera}-defaults.csv"],
                             "analysis_paths": ["analysis_paths-{machine}-{camera}-defaults.json"],
                             "surface_props": ["surface_props-{machine}-{camera}-defaults.json"]}
    if params is None:
        params = {}
    params.update({'pulse': pulse, 'camera': camera, 'machine': machine, 'fire_path': fire_paths['root']})

    files = {}
    # Locate lookup files and info for pulse
    lookup_files = ['calcam_calibs', 'analysis_paths']
    lookup_info = {}
    for file_type in lookup_files:
        path, fn, info = lookup_pulse_info(pulse, camera, machine, params=params,
                                           search_paths=search_paths_inputs, filename_patterns=fn_patterns_inputs[file_type])
        files[f'{file_type}_lookup'] = path / fn
        lookup_info[file_type] = info

    # Get filenames referenced from lookup files: Calcam calibration file
    lookup_references = [['calcam_calib', 'calcam_calibs', 'calcam_calibration_file'],]
                   # ['analysis_path_dfn', 'analysis_path_dfns', 'analysis_path_name']]
    for name, file_type, column in lookup_references:
        calcam_calib_fn = lookup_info[file_type][column]
        path, fn = locate_file(search_paths_inputs, calcam_calib_fn, path_kws=params, fn_kws=params)
        path_fn = path / fn
        files[name] = path_fn
        if not path_fn.is_file():
            raise IOError(f'Required input file "{path_fn}" does not exist')

    # Get filenames straight from config settings: Analysis path definition file
    input_files = ['analysis_path_dfns', 'black_body_curve']
    for input_file in input_files:
        fn_patterns = fn_patterns_inputs[input_file]
        path, fn = locate_file(search_paths_inputs, fn_patterns, path_kws=params, fn_kws=params)
        path_fn = path / fn
        files[input_file] = path_fn

    # Black body calibration file

    # Surface properties file

    # Checkpoint intermediate output files to speed up analysis
    checkpoint_path = setup_checkpoint_path(paths_output['checkpoint_data'])
    files['checkpoint_path'] = checkpoint_path
    params['calcam_calib_stem'] = files['calcam_calib'].stem

    # Calcam raycast checkpoint
    checkpoints = ['raycast_checkpoint']
    for checkpoint in checkpoints:
        raycast_checkpoint_fn = fn_pattern_output[checkpoint].format(**params)
        checkpoint_path_fn = checkpoint_path / checkpoint / raycast_checkpoint_fn
        files[checkpoint] = checkpoint_path_fn
        if not checkpoint_path_fn.parent.is_dir():
            checkpoint_path_fn.parent.mkdir()

    # TODO: check path characters are safe (see setpy datafile code)

    return files, lookup_info

def setup_checkpoint_path(path: Union[str, Path]):
    path = format_path(path)
    if not path.exists():
        path.mkdir(parents=True)
        logger.info(f'Created fire checkpoint data directory: {path}')
    return path

def try_movie_plugins(plugin_key, pulse, camera, machine, movie_plugins, movie_paths=None, movie_fns=None):
    kwargs = {'machine': machine, 'camera': camera, 'pulse': pulse, 'pulse_prefix': str(pulse)[0:2]}
    data, origin = None, None
    for plugin_name, movie_plugin in movie_plugins.items():
        read_movie_func = movie_plugin[plugin_key]
        plugin_info = movie_plugin['plugin_info']
        signature = inspect.signature(read_movie_func).parameters.keys()
        if ("path_fn" in signature):
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
        raise IOError(f'Failed to read movie data for args: {kwargs} with movie plugins: {list(movie_plugins.keys())}')
    return data, origin

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
    plugin_key = 'meta'
    meta_data, origin = try_movie_plugins(plugin_key, pulse, camera, machine, movie_plugins,
                                          movie_paths=movie_paths, movie_fns=movie_fns)
    return meta_data, origin

def read_movie_data(pulse: Union[int, str], camera: str, machine: str, movie_plugins: dict,
                    movie_paths: Optional[PathList]=None, movie_fns: Optional[Sequence[str]]=None) \
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
    return frame_nos, frame_times, frame_data, origin

def generate_pulse_id_strings(id_strings, pulse, camera, machine, pass_no=0):
    """Return standardised ID strings used for consistency in filenames and data labels
    :param id_strings: Dict of string_ids to update/populate
    :param pulse: Shot/pulse number or string name for synthetic movie data
    :param camera: Name of camera to analyse (unique name of camera or diagnostic code)
    :param machine: Tokamak that the data originates from
    :return: Dict of ID strings
    """
    pulse_id = f'{machine}-{pulse}'
    camera_id = f'{pulse_id}-{camera}'
    pass_id = f'{pulse_id}-{pass_no}'

    # calcam_id = f'{machine}-{camera}-{calib_date}-{pass_no}'

    id_strings.update({'pulse_id': pulse_id,
                       'camera_id': camera_id,
                       'pass_id': pass_id})

    return id_strings

def generate_camera_id_strings(id_strings, lens, t_int):
    """Return standardised ID strings used for consistency in filenames and data labels

    :param id_strings: Dict of string_ids to update/populate
    :param lens: Lens on camera
    :param t_int: Integration time of camera in seconds
    :return: Dict of ID strings
    """
    camera_id = id_strings['camera_id']
    lens_id = f'{camera_id}-{lens}'
    t_int_id = f'{lens_id}-{t_int}'

    id_strings.update({'lens_id': lens_id,
                       't_int_id': t_int_id, })
    return id_strings

def generate_frame_id_strings(id_strings, frame_no, frame_time):
    """Return ID strings for specific analysis frame

    :param id_strings: Dict of string_ids to update/populate
    :param frame_no: Frame number from movie meta data
    :param frame_time: Frame time from movie meta data
    :return: Dict of ID strings
    """
    camera_id = id_strings['camera_id']
    pulse_id = id_strings['pulse_id']

    frame_id = f'{camera_id}-{frame_no}'
    time_id = f'{pulse_id}-{frame_time}'

    id_strings.update({'frame_id': frame_id,
                       'time_id': time_id, })
    return id_strings


def check_frame_range(meta_data, frames=None, start_frame=None, end_frame=None, nframes_user=None, frame_stride=1):
    raise NotImplementedError

def json_dump(obj, path_fn, fn=None, indent=4, overwrite=True, raise_on_fail=True):
    path_fn = Path(path_fn)
    if fn is not None:
        path_fn = path_fn / fn
    if (not overwrite) and (path_fn.exists()):
        raise FileExistsError(f'Requested json file already exists: {path_fn}')
    try:
        with open(path_fn, 'w') as f:
            json.dump(obj, f, indent=indent)
        out = path_fn
    except Exception as e:
        out = e
        if raise_on_fail:
            raise e
    return out

# def json_load(path_fn: Union[str, Path], fn: Optional(str)=None, keys: Optional[Sequence[str]]=None):
def json_load(path_fn, fn=None, keys=None):
    """Read json file with optional indexing

    Args:
        path_fn : Path to json file
        fn      : Optional filename to append to path
        keys    : Optional keys to subset of contents to return

    Returns: Contents of json file

    """
    path_fn = Path(path_fn)
    if fn is not None:
        path_fn = path_fn / fn
    if not path_fn.exists():
        raise FileNotFoundError(f'Requested json file does not exist: {path_fn}')
    try:
        with open(path_fn, 'r') as f:
            contents = json.load(f)
    except Exception as e:
        raise e
    out = contents
    # Return indexed subset of file
    if keys is not None:
        keys = make_iterable(keys)
        for key in keys:
            try:
                out = out[key]
            except KeyError as e:
                raise KeyError(f'json file ({path_fn}) does not contain key "{key}" in {out}')
    return out

def two_level_dict_to_multiindex_df(d):
    """Convert nested dictionary to two level multiindex dataframe

    Args:
        d: Input nested dictionary

    Returns: DataFrame containing contents of d

    Examples:
        d:
        {"MAST_S1_L3_centre_radial_1":
            {
              "start": {"R": 0.769, "z": -1.827, "phi": 70.6},
              "end": {"R": 1.484, "z": -1.831, "phi": 70.6}
            }
        }
        df:
                                                  R      z   phi
            MAST_S1_L3_centre_radial_1 end    1.484 -1.831  70.6
                                       start  0.769 -1.827  70.6

    """
    df = pd.DataFrame.from_dict({(k1, k2): v2 for k1, v1 in d.items() for k2, v2 in v1.items()}, orient='index')
    return df


def lookup_pulse_row_in_csv(path_fn: Union[str, Path], pulse: int) -> Union[pd.Series, Exception]:
    """Return row from csv file containing information for pulse range containing supplied pulse number

    :param path_fn: path to csv file containing pulse range information
    :param pulse: pulse number of interest
    :return: Pandas Series containing pulse information / Exception if unsuccessful
    """
    try:
        table = pd.read_csv(path_fn)
    except FileNotFoundError:
        return FileNotFoundError(f'Calcam calib lookup file: {path_fn}')
    if not np.all([col in list(table.columns) for col in ['pulse_start', 'pulse_end']]):
        raise IOError(f'Unsupported pulse row CSV file format - '
                      f'must contain "pulse_start", "pulse_end" columns: {path_fn}')
    row_mask = np.logical_and(table['pulse_start'] <= pulse, table['pulse_end'] >= pulse)
    if np.sum(row_mask) > 1:
        raise ValueError(f'Calcam calib lookup file contains overlapping ranges. Please fix: {path_fn}')
    elif np.sum(row_mask) == 0:
        pulse_ranges = list(zip(table['pulse_start'], table['pulse_end']))
        return ValueError(f'Pulse {pulse} does not fall in any pulse range {pulse_ranges} in {path_fn}')
    else:
        calib_info = table.loc[row_mask].iloc[0]
    return calib_info


def lookup_pulse_info(pulse: Union[int, str], camera: str, machine: str, search_paths: PathList,
                          filename_patterns: Union[str, Path], params: Optional[dict]=None, raise_=True) -> pd.Series:
    params = {} if params is None else params
    params.update({'pulse': pulse, 'camera': camera, 'machine': machine})
    path, fn = locate_file(search_paths, fns=filename_patterns, path_kws=params, fn_kws=params)
    info = lookup_pulse_row_in_csv(path/fn, pulse)
    if raise_ and isinstance(info, Exception):
        raise info
    return path, fn, info

def check_settings_complete(settings, machine, camera):
    """Check settings contain required information

    Raises exceptions if required information is missing

    Args:
        settings:   Settings loaded from fire config file
        machine:    Machine for analysis
        camera:     Camera for analysis

    Returns: None

    """
    sub_settings = settings['machines']
    if machine not in sub_settings:
        raise ValueError(f'Fire settings do not contain settings for machine: "{machine}"\n{sub_settings}')
    sub_settings = settings['machines'][machine]['cameras']
    if camera not in sub_settings:
        raise ValueError(f'Fire settings do not contain settings for camera: "{camera}"\n{sub_settings}')

def get_compatible_movie_plugins(settings, machine, camera):
    plugin_paths = settings['paths_input']['movie_plugins']
    plugin_paths = [p.format(fire_path=fire_paths['root']) for p in plugin_paths]
    movie_plugins_all = get_movie_plugins(plugin_paths)
    movie_plugins_compatible = settings['machines'][machine]['cameras'][camera]['movie_plugins']
    movie_plugins_compatible = {key: value for key, value in movie_plugins_all.items()
                                    if key in movie_plugins_compatible}
    return movie_plugins_compatible

def get_movie_plugins(plugin_paths):
    plugin_attributes = ['movie_plugin_name', 'read_movie_meta', 'read_movie_data', 'plugin_info']
    plugins = search_for_plugins(plugin_paths, plugin_attributes)
    plugins = {objs[0]: {'meta': objs[1], 'data': objs[2], 'plugin_info': objs[3]} for mod, objs in plugins}
    logger.info(f'Located movie plugins for: {", ".join(list(plugins.keys()))}')
    return plugins

def search_for_plugins(plugin_paths, plugin_attributes):

    plugins_all = []
    plugin_attributes = make_iterable(plugin_attributes)

    plugin_paths = make_iterable(plugin_paths)
    for path in plugin_paths:
        plugins = get_plugins(path, plugin_attributes)
        plugins_all += plugins
    return plugins_all

def get_plugins(path, plugin_attributes):
    plugins = []
    file_list = glob.glob(os.path.join(path, '*'))
    # Get possible modules in directory
    possible_plugin_modules = []
    for f in file_list:
        if os.path.isdir(f) and os.path.isfile(os.path.join(f, '__init__.py')):
            possible_plugin_modules.append(os.path.join(f, '__init__.py'))
        elif f.endswith('.py'):
            possible_plugin_modules.append(f)
    # TODO: Handle empty trylist
    for path_fn in possible_plugin_modules:
        module = get_module_from_path_fn(path_fn)
        if module is None:
            continue
        plugin = [module, []]
        for attribute in plugin_attributes:
            try:
                plugin[1].append(getattr(module, attribute))
            except AttributeError as e:
                if len(plugin[1]) > 0:
                    logger.warning(f'Cannot load incomplete plugin {plugin} missing "{attribute}"')
                else:
                    pass
                break
        else:
            plugins.append(plugin)
    # plugin = ([module_name, path_fn, ''.join(traceback.format_exception_only(sys.exc_info()[0], sys.exc_info()[1]))])

    return plugins

def get_module_from_path_fn(path_fn):
    path_fn = Path(path_fn)
    module = None
    if path_fn.exists():
        if path_fn.name == '__init__.py':
            module_name = os.sep.join(path_fn.parts[-3:-1])
        else:
            module_name = os.sep.join(path_fn.parts[-2:])
        try:
            spec = importlib.util.spec_from_file_location(module_name, path_fn)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            module = None
    return module
