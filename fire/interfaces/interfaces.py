# -*- coding: future_fstrings -*-
#!/usr/bin/env python
"""
The `interfaces` module contains functions for interfacing with other codes and files.
"""

import os, logging, json
import importlib.util
from typing import Union, Sequence, Optional
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd

from fire.utils import locate_file, make_iterable
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
    params.update({'pulse': pulse, 'camera': camera, 'machine': machine, 'fire_path': str(fire_paths['root'])})

    files = {}
    # Locate lookup files and info for pulse
    lookup_files = ['calcam_calibs', 'analysis_paths']
    lookup_info = {}
    for file_type in lookup_files:
        path, fn, info = lookup_pulse_info(pulse, camera, machine, params=params,
                                           search_paths=search_paths_inputs, filename_patterns=fn_patterns_inputs[file_type],
                                           file_type=file_type)
        files[f'{file_type}_lookup'] = path / fn
        lookup_info[file_type] = info

    # Get filenames referenced from lookup files: Calcam calibration file
    lookup_references = [['calcam_calib', 'calcam_calibs', 'calcam_calibration_file'],]
    # ['analysis_path_dfn', 'analysis_path_dfns', 'analysis_path_name']]
    for name, file_type, column in lookup_references:
        calcam_calib_fn = lookup_info[file_type][column]
        try:
            path, fn = locate_file(search_paths_inputs, calcam_calib_fn, path_kws=params, fn_kws=params)
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Could not locate fire input file for "{file_type}":\n{str(e)}')
        path_fn = path / fn
        files[name] = path_fn
        if not path_fn.is_file():
            raise IOError(f'Required input file "{path_fn}" does not exist')

    # Get filenames straight from config settings: Analysis path definition file, bb photons, temperature coefs,
    # surface coords, surface properties
    input_files = ['analysis_path_dfns', 'black_body_curve', 'calib_coefs', 'structure_coords', 'material_props']
    for file_type in input_files:
        fn_patterns = fn_patterns_inputs[file_type]
        try:
            path, fn = locate_file(search_paths_inputs, fn_patterns, path_kws=params, fn_kws=params)
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Could not locate fire input file for "{file_type}":\n{str(e)}')
        path_fn = path / fn
        files[file_type] = path_fn

    # Checkpoint intermediate output files to speed up analysis
    # checkpoint_path = setup_checkpoint_path(paths_output['checkpoint_data'])
    checkpoint_path = Path(paths_output['checkpoint_data']).expanduser().resolve()
    files['checkpoint_path'] = checkpoint_path
    params['calcam_calib_stem'] = files['calcam_calib'].stem

    # Calcam raycast checkpoint
    checkpoints = ['raycast_checkpoint']
    for checkpoint in checkpoints:
        checkpoint_fn = fn_pattern_output[checkpoint].format(**params)
        checkpoint_path_fn = checkpoint_path / checkpoint / checkpoint_fn
        files[checkpoint] = checkpoint_path_fn
        checkpoint_path_fn.parent.mkdir(parents=True, exist_ok=True)

    # Output filenames
    outputs = ['processed_ir_netcdf']
    for output in outputs:
        output_fn = fn_pattern_output[output].format(**params)
        # TODO: Set output path with config file/run argument?
        files[output] = Path('.') / output_fn

    # TODO: check path characters are safe (see setpy datafile code)

    return files, lookup_info

# def setup_checkpoint_path(path: Union[str, Path]):
#     path = format_path(path)
#     if not path.exists():
#         path.mkdir(parents=True)
#         logger.info(f'Created fire checkpoint data directory: {path}')
#     return path

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

def json_dump(obj, path_fn: Union[str, Path], path: Optional[Union[str, Path]]=None, indent: int=4,
              overwrite: bool=True, raise_on_fail: bool=True):
    """Convenience wrapper for json.dump.

    Args:
        obj             : Object to be serialised
        path_fn         : Filename (and path) for output file
        path            : (Optional) path to output file
        indent          : Number of spaces for each json indentation
        overwrite       : Overwite existing file
        raise_on_fail   : Whether to raise exceptions or return them

    Returns: Output file path if successful, else captured exception

    """
    path_fn = Path(path_fn)
    if path is not None:
        path_fn = Path(path) / path_fn
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

# def json_load(path_fn: Union[str, Path], fn: Optional(str)=None, keys: Optional[Sequence[Sequence]]=None):
def json_load(path_fn: Union[str, Path], path: Optional[Union[str, Path]]=None,
              key_paths: Optional[Sequence[Sequence]]=None, lists_to_arrays: bool=False,
              raise_on_filenotfound: bool=True):
    """Read json file with optional indexing

    Args:
        path_fn         : Path to json file
        path            : Optional path to prepend to filename
        key_paths       : Optional keys to subsets of contents to return. Each element of keys should be an iterable
                          specifiying a key path through the json file.
        lists_to_arrays : Whether to cast lists in output to arrays for easier slicing etc.
        raise_on_filenotfound : Whether to raise (or else return) FileNotFoundError if file not located

    Returns: Contents of json file

    """
    path_fn = Path(path_fn)
    if path is not None:
        path_fn = Path(path) / path_fn
    if not path_fn.exists():
        e = FileNotFoundError(f'Requested json file does not exist: {path_fn}')
        if raise_on_filenotfound:
            raise e
        else:
            return e
    try:
        with open(str(path_fn), 'r') as f:
            contents = json.load(f)
    except Exception as e:
        raise e
    # Return indexed subset of file
    if key_paths is not None:
        key_paths = make_iterable(key_paths)
        out = {}
        for key_path in key_paths:
            key_path = make_iterable(key_path)
            subset = contents
            for key in key_path:
                try:
                    subset = subset[key]
                except KeyError as e:
                    raise KeyError(f'json file ({path_fn}) does not contain key "{key}" in key path "{key_path}":\n'
                                   f'{subset}')
            out[key_path[-1]] = subset

    else:
        out = contents
    if lists_to_arrays:
        out = cast_lists_in_dict_to_arrays(out)
    return out

def cast_lists_in_dict_to_arrays(dict_in):
    dict_out = deepcopy(dict_in)
    for key, value in dict_out.items():
        if isinstance(value, (list)):
            dict_out[key] = np.array(value)
        elif isinstance(value, dict):
            dict_out[key] = cast_lists_in_dict_to_arrays(value)

    return dict_out

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


def lookup_pulse_row_in_csv(path_fn: Union[str, Path], pulse: int, **kwargs_csv) -> Union[pd.Series, Exception]:
    """Return row from csv file containing information for pulse range containing supplied pulse number

    :param path_fn: path to csv file containing pulse range information
    :param pulse: pulse number of interest
    :return: Pandas Series containing pulse information / Exception if unsuccessful
    """
    table = read_csv(path_fn, **kwargs_csv)
    if isinstance(table, Exception):
        calib_info = table
    else:
        if not np.all([col in list(table.columns) for col in ['pulse_start', 'pulse_end']]):
            raise IOError(f'Unsupported pulse row CSV file format - '
                          f'must contain "pulse_start", "pulse_end" columns: {path_fn}')
        table = table.astype({'pulse_start': int, 'pulse_end': int})
        row_mask = np.logical_and(table['pulse_start'] <= pulse, table['pulse_end'] >= pulse)
        if np.sum(row_mask) > 1:
            raise ValueError(f'Calcam calib lookup file contains overlapping ranges. Please fix: {path_fn}')
        elif np.sum(row_mask) == 0:
            pulse_ranges = list(zip(table['pulse_start'], table['pulse_end']))
            return ValueError(f'Pulse {pulse} does not fall in any pulse range {pulse_ranges} in {path_fn}')
        else:
            calib_info = table.loc[row_mask].iloc[0]
    return calib_info

def read_csv(path_fn: Union[Path, str], **kwargs):
    path_fn = Path(path_fn)
    if 'sep' not in kwargs:
        if path_fn.suffix == '.csv':
            kwargs['sep'] = ','
        elif path_fn.suffix == '.tsv':
            kwargs['sep'] = r'\s+'
        else:
            kwargs['sep'] = None  # Use csv.Sniffer tool
    try:
        table = pd.read_csv(path_fn, **kwargs)  # index_col
    except FileNotFoundError:
        return FileNotFoundError(f'CSV file does not exist: {path_fn}')
    return table

def lookup_pulse_info(pulse: Union[int, str], camera: str, machine: str, search_paths: PathList,
                          filename_patterns: Union[str, Path], params: Optional[dict]=None,
                          file_type: Optional[str]=None, raise_=True) -> pd.Series:
    """Extract information from pulse look up file

    Args:
        pulse               : Shot/pulse number or string name for synthetic movie data
        camera              : Name of camera to analyse (unique name of camera or diagnostic code)
        machine             : Tokamak that the data originates from
        search_paths        : Format strings for paths to search for files
        filename_patterns   : Format string for possible filesnames to locate
        params              : Parameters to substitute into format strings
        raise_              : Whether to raise or return exceptions

    Returns: Series of data from csv file

    """
    params = {} if params is None else params
    params.update({'pulse': pulse, 'camera': camera, 'machine': machine})
    try:
        path, fn = locate_file(search_paths, fns=filename_patterns, path_kws=params, fn_kws=params, raise_=True)
    except FileNotFoundError as e:
        message = (f'Failed to locate "{file_type}" pulse lookup file for '
                   f'machine="{machine}", camera="{camera}", pulse="{pulse}"\n{str(e)}')
        if raise_:
            raise FileNotFoundError(message)
        else:
            return None, None, message
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


def get_module_from_path_fn(path_fn):
    path_fn = Path(path_fn)
    module = None
    if path_fn.exists():
        if path_fn.name == '__init__.py':
            module_name = os.sep.join(path_fn.parts[-3:-1])
        else:
            module_name = os.sep.join(path_fn.parts[-2:])
        try:
            spec = importlib.util.spec_from_file_location(module_name, str(path_fn))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            logger.exception('Failed to load potential plugin module: {path_fn}'.format(path_fn=path_fn))
            module = None
    return module
