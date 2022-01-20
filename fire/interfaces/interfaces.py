# -*- coding: future_fstrings -*-
#!/usr/bin/env python
"""
The `interfaces` module contains functions for interfacing with other codes and files.
"""

import os, logging, json, re
import importlib.util
from typing import Union, Sequence, Optional
from pathlib import Path

import numpy as np
import pandas as pd

from fire.interfaces.io_basic import read_csv
from fire.misc.utils import (locate_file, mkdir, filter_nested_dict_key_paths, drop_nested_dict_key_paths, cast_lists_to_arrays)
from fire.interfaces.exceptions import InputFileException

logger = logging.getLogger(__name__)
# logger.propagate = False
# logger.setLevel(logging.INFO)
# print(logger_info(logger))

PathList = Sequence[Union[Path, str]]

def digest_shot_file_name(fn, pattern='(?P<diag_tag>\D*)(?P<shot>\d+)\.(?P<extension>\w+)'):
    info = {}
    fn = str(Path(fn).name)
    m = re.match(pattern, fn)
    if m:
        info = m.groupdict()
        try:
            info['shot'] = int(info['shot'])
        except (KeyboardInterrupt, ValueError) as e:
            pass
    return info

def format_path(path: Union[str, Path], base_paths: Union[dict, str]=(), **kwargs) -> Path:
    kwargs.update(dict(base_paths))
    path = Path(str(path).format(**kwargs)).expanduser()
    return path

def identify_files(pulse, camera, machine, search_paths_inputs=None, fn_patterns_inputs=None,
                   paths_output=None, base_paths=(), fn_pattern_checkpoints=None, params=None):
    """Return dict of paths to input files needed for IR analysis

    :param pulse: Shot/pulse number or string name for synthetic movie data
    :param camera: Name of camera to analyse (unique name of camera or diagnostic code)
    :param machine: Tokamak that the data originates from
    :return: Dict of filepaths
    """
    if search_paths_inputs is None:
        search_paths_inputs = ["~/fire/input_files/{machine}/", "{fire_source_dir}/input_files/{machine}/", "~/calcam/calibrations/"]
    if fn_patterns_inputs is None:
        # TODO: UPDATE
        fn_patterns_inputs = {"calcam_calibs": ["calcam_calibs-{machine}-{diag_tag_raw}-defaults.csv"],
                              "analysis_paths": ["analysis_paths-{machine}-{diag_tag_raw}-defaults.json"],
                              "surface_props": ["surface_props-{machine}-{diag_tag_raw}-defaults.json"]}
    if params is None:
        params = {}
    base_paths = dict(base_paths)

    params.update({'pulse': pulse, 'camera': camera, 'machine': machine, **base_paths})

    files = {}

    # Locate lookup files and extract relevant info for specific pulse - this info may be a filename for lookup at
    # next step
    lookup_files = ['camera_settings', 'calcam_calibs', 'analysis_paths', 'temperature_coefs']
    lookup_info = {}
    for file_type in lookup_files:
        path, fn, info = lookup_pulse_info(pulse, camera, machine, params=params,
                                           search_paths=search_paths_inputs, filename_patterns=fn_patterns_inputs[file_type],
                                           file_type=file_type, csv_kwargs={'comment': '#'})
        # TODO: consider moving file comment type to fire_config?
        files[f'{file_type}_lookup'] = path / fn
        lookup_info[file_type] = info

    # Get filenames referenced from lookup files: Calcam calibration file
    lookup_references = [
                            ['calcam_calib', 'calcam_calibs', 'calcam_calibration_file'],  # name, file_type, column
                        ]
    # ['analysis_path_dfn', 'analysis_path_dfns', 'analysis_path_name']]
    for name, file_type, column in lookup_references:
        fn_pulse = lookup_info[file_type][column]
        try:
            path, fn = locate_file(search_paths_inputs, fn_pulse, path_kws=params, fn_kws=params)
        except FileNotFoundError as e:
            raise FileNotFoundError(f'Could not locate fire input file for "{file_type}":\n{str(e)}')
        path_fn = path / fn
        files[name] = path_fn
        if not path_fn.is_file():
            raise IOError(f'Required input file "{path_fn}" does not exist')

    # Get filenames straight from config settings: Analysis path definition file, bb photons,
    # surface coords, surface properties
    input_files = ['analysis_path_dfns',
                   # 'black_body_curve',
                   'structure_coords', 'material_props']
    errors = {}
    for file_type in input_files:
        fn_patterns = fn_patterns_inputs[file_type]
        try:
            path, fn = locate_file(search_paths_inputs, fn_patterns, path_kws=params, fn_kws=params)
        except FileNotFoundError as e:
            err = FileNotFoundError(f'Could not locate fire input file for "{file_type}":\n{str(e)}')
            logger.warning(err)
            errors[file_type] = err
        else:
            path_fn = path / fn
            files[file_type] = path_fn
    if len(errors) > 0:
        raise FileNotFoundError(f'Failed to locate {len(errors)} files for: {list(errors.keys())}. First error:\n'
                                f'{errors[list(errors.keys())[0]]}')
    # Checkpoint intermediate output files to speed up analysis
    # checkpoint_path = setup_checkpoint_path(paths_output['checkpoint_data'])
    checkpoint_path = Path(paths_output['checkpoint_data']).expanduser().resolve()
    files['checkpoint_path'] = checkpoint_path
    params['calcam_calib_stem'] = files['calcam_calib'].stem

    # Calcam raycast checkpoint
    checkpoints = ['raycast']
    for checkpoint in checkpoints:
        checkpoint_fn = fn_pattern_checkpoints[checkpoint].format(**params)
        checkpoint_path_fn = checkpoint_path / checkpoint / checkpoint_fn
        files[checkpoint+'_checkpoint'] = checkpoint_path_fn
        checkpoint_path_fn.parent.mkdir(parents=True, exist_ok=True)

    # Output filenames
    # outputs = ['processed_ir_netcdf']
    # for output in outputs:
    #     output_fn = fn_pattern_output[output].format(**params)
    #     # TODO: Set output path with config file/run argument?
    #     files[output] = Path('.') / output_fn

    # TODO: check path characters are safe (see setpy datafile code)

    return files, lookup_info

# def setup_checkpoint_path(path: Union[str, Path]):
#     path = format_path(path)
#     if not path.exists():
#         path.mkdir(parents=True)
#         logger.info(f'Created fire checkpoint data directory: {path}')
#     return path

def generate_pulse_id_strings(id_strings, pulse, diag_tag_raw, machine, pass_no=0):
    """Return standardised ID strings used for consistency in filenames and data labels
    :param id_strings: Dict of string_ids to update/populate
    :param pulse: Shot/pulse number or string name for synthetic movie data
    :param camera: Name of camera to analyse (unique name of camera or diagnostic code)
    :param machine: Tokamak that the data originates from
    :return: Dict of ID strings
    """
    pulse_id = f'{machine}-{pulse}'
    camera_id = f'{pulse_id}-{diag_tag_raw}'
    pass_id = f'{pulse_id}-{pass_no}'

    # calcam_id = f'{machine}-{diag_tag_raw}-{calib_date}-{pass_no}'

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

# def json_dump(obj, path_fn: Union[str, Path], path: Optional[Union[str, Path]]=None, indent: int=4,
#               overwrite: bool=True, raise_on_fail: bool=True):
#     """Convenience wrapper for json.dump.
#
#     Args:
#         obj             : Object to be serialised
#         path_fn         : Filename (and path) for output file
#         path            : (Optional) path to output file
#         indent          : Number of spaces for each json indentation
#         overwrite       : Overwite existing file
#         raise_on_fail   : Whether to raise exceptions or return them
#
#     Returns: Output file path if successful, else captured exception
#
#     """
#     # TODO: Convert ndarrays to lists so json serialisable
#     path_fn = Path(path_fn)
#     if path is not None:
#         path_fn = Path(path) / path_fn
#     if (not overwrite) and (path_fn.exists()):
#         raise FileExistsError(f'Requested json file already exists: {path_fn}')
#     try:
#         with open(path_fn, 'w') as f:
#             json.dump(obj, f, indent=indent)
#         out = path_fn
#     except Exception as e:
#         out = e
#         if raise_on_fail:
#             raise e
#     return out

# def json_load(path_fn: Union[str, Path], fn: Optional(str)=None, keys: Optional[Sequence[Sequence]]=None):
def json_load(path_fn: Union[str, Path], path: Optional[Union[str, Path]]=None,
              key_paths_keep: Optional[Sequence[Sequence]]=None, key_paths_drop=('README',),
              compress_key_paths=True, lists_to_arrays: bool=False, raise_on_filenotfound: bool=True):
    """Read json file with optional indexing

    Args:
        path_fn         : Path to json file
        path            : (Optional) path to prepend to filename
        key_paths_keep  : (Optional) keys to subsets of contents to return. Each element of keys should be an iterable
                          specifiying a key path through the json file.
        lists_to_arrays : (Bool) Whether to cast lists in output to arrays for easier slicing etc.
        raise_on_filenotfound : (Bool) Whether to raise (or else return) FileNotFoundError if file not located

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
    except json.decoder.JSONDecodeError as e:
        # raise InputFileException(original_exception=e, info={'fn': path_fn})
        raise InputFileException(f'Invalid json formatting in input file "{path_fn}"', e)

    # Return indexed subset of file
    out = filter_nested_dict_key_paths(contents, key_paths_keep=key_paths_keep,
                                       compress_key_paths=compress_key_paths, path_fn=path_fn)

    # Drop some keys
    out = drop_nested_dict_key_paths(out, key_paths_drop=key_paths_drop)

    if lists_to_arrays:
        out = cast_lists_to_arrays(out)

    return out


def lookup_pulse_row_in_csv(path_fn: Union[str, Path], pulse: int, allow_overlaping_ranges: bool=False,
                            raise_exceptions: bool=True, **kwargs_csv) -> Union[pd.Series, Exception]:
    """Return row from csv file containing information for pulse range containing supplied pulse number

    :param path_fn: path to csv file containing pulse range information
    :param pulse: pulse number of interest
    :return: Pandas Series containing pulse information / Exception if unsuccessful
    """
    table = read_csv(path_fn, python_types_kwargs=dict(list_delimiters=(',', ' ')), **kwargs_csv)

    if isinstance(table, Exception):
        pulse_info = table
    else:
        pulse_info = lookup_pulse_row_in_df(table, pulse=pulse, allow_overlaping_ranges=allow_overlaping_ranges,
                                            description=path_fn, raise_=raise_exceptions)
    if isinstance(pulse_info, Exception) and raise_exceptions:
        raise pulse_info
    else:
        return pulse_info

def lookup_pulse_row_in_df(df: pd.DataFrame, pulse: int, allow_overlaping_ranges: bool=False,
                            description=None, raise_: bool=True, **kwargs_csv) -> Union[pd.Series, Exception]:

    if not np.all([col in list(df.columns) for col in ['pulse_start', 'pulse_end']]):
        pulse_info = IOError(f'Unsupported pulse row CSV file format - '
                      f'must contain "pulse_start", "pulse_end" columns: {path_fn}')
    else:
        # TODO: Allow None for eg end of pulse range if current value
        table = df.astype({'pulse_start': int, 'pulse_end': int})
        row_mask = np.logical_and(table['pulse_start'] <= pulse, table['pulse_end'] >= pulse)
        if (np.sum(row_mask) > 1) and (not allow_overlaping_ranges):
            pulse_info = ValueError(f'Lookup file {path_fn} contains overlapping ranges. Please fix: \n'
                                    f'{table.loc[row_mask]}')
        elif np.sum(row_mask) == 0:
            pulse_ranges = list(zip(table['pulse_start'], table['pulse_end']))
            pulse_info = ValueError(f'Pulse {pulse} does not fall in any pulse range {pulse_ranges} in {description}')
        else:
            pulse_info = table.loc[row_mask]
            if np.sum(row_mask) == 1:
                pulse_info = pulse_info.iloc[0]
    return pulse_info



def lookup_pulse_info(pulse: Union[int, str], diag_tag_raw: str, machine: str, search_paths: PathList,
                      filename_patterns: Union[str, Path], params: Optional[dict]=None,
                      file_type: Optional[str]=None,
                      csv_kwargs: Optional[dict]=None, raise_=True) -> pd.Series:
    """Extract information from pulse look up file

    Args:
        pulse               : Shot/pulse number or string name for synthetic movie data
        diag_tag_raw              : Name of camera to analyse (unique name of camera or diagnostic code)
        machine             : Tokamak that the data originates from
        search_paths        : Format strings for paths to search for files
        filename_patterns   : Format string for possible filesnames to locate
        params              : Parameters to substitute into format strings
        csv_kwargs          : Keyword arguments to pass to (pandas) csv reader
        raise_              : Whether to raise or return exceptions

    Returns: Series of data from csv file

    """
    params = {} if params is None else params
    csv_kwargs = {} if csv_kwargs is None else csv_kwargs
    params.update({'pulse': pulse, 'camera': diag_tag_raw, 'diag_tag_raw': diag_tag_raw, 'machine': machine})
    try:
        path, fn = locate_file(search_paths, fns=filename_patterns, path_kws=params, fn_kws=params, raise_=True)
    except FileNotFoundError as e:
        message = (f'Failed to locate "{file_type}" pulse lookup file for '
                   f'machine="{machine}", camera="{diag_tag_raw}", pulse="{pulse}"\n{str(e)}')
        if raise_:
            raise FileNotFoundError(message)
        else:
            return None, None, message
    info = lookup_pulse_row_in_csv(path / fn, pulse, raise_exceptions=False, **csv_kwargs)
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
    # TODO: Add path to config file to error messages
    sub_settings = settings['machines']
    if machine not in sub_settings:
        raise ValueError(f'Fire config settings file does not contain settings for machine "{machine}". '
                         f'Options: {", ".join(list(sub_settings.keys()))}')
    sub_settings = settings['machines'][machine]['cameras']
    if camera not in sub_settings:
        # TODO: include alias options
        raise ValueError(f'Fire config settings file does not contain settings for "{machine}" camera "{diag_tag_raw}". '
                         f'Options: {", ".join(list(sub_settings.keys()))}')



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

def archive_netcdf_output(path_fn_in, path_archive='~/{fire_user_dir}/archive_netcdf_output/{diag_tag_raw}/',
                          meta_data=None):
    success = False
    if path_fn_in is None:
        return success

    if meta_data is None:
        meta_data = {}

    path_archive = path_archive.format(**meta_data)

    path_fn_in = Path(path_fn_in)
    fn = path_fn_in.name

    path_archive = Path(path_archive).expanduser()  # fire_paths[]
    mkdir(path_archive, depth=2)
    path_fn_archive = path_archive / fn

    if path_fn_in.exists():
        path_fn_archive.write_bytes(path_fn_in.read_bytes())
        logger.info(f'Copied netcdf output file "{path_fn_archive}" from "{path_fn_in}"')

    return success
