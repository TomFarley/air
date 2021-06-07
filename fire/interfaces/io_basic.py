#!/usr/bin/env python

"""Basic I/O functions separated from io_utils to avoid import loops when used in top level __init__.py


Created: Tom Farley, 21-04-21
"""

import logging
import os
import pickle
import re
import warnings
from typing import Union, Iterable, Sequence, Tuple, Optional, Any, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from fire.interfaces.exceptions import InputFileException
from fire.misc import utils
from fire.misc.utils import (compare_dict, convert_dataframe_values_to_python_types, ask_input_yes_no,
    get_traceback_location, make_iterable)

logger = logging.getLogger(__name__)
logger.propagate = False


def test_pickle(obj, verbose=True):
    """Test if an object can successfully be pickled and loaded again
    Returns True if succeeds
            False if fails
    """
    import pickle
    # sys.setrecursionlimit(10000)
    path = 'test_tmp.p.tmp'
    if os.path.isfile(path):
        os.remove(path)  # remove temp file
    try:
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        if verbose:
            print('Pickled object')
        with open(path, 'rb') as f:
            out = pickle.load(f)
        if verbose:
            print('Loaded object')
    except Exception as e:
        if verbose:
            print('{}'.format(e))
        return False
    if os.path.isfile(path):
        if verbose:
            print('Pickled file size: {:g} Bytes'.format(os.path.getsize(path)))
        os.remove(path)  # remove temp file
    import pdb; pdb.set_trace()
    if verbose:
        print('In:\n{}\nOut:\n{}'.format(out, obj))
    if not isinstance(obj, dict):
        out = out.__dict__
        obj = obj.__dict__
    if compare_dict(out, obj):
        return True
    else:
        return False


def pickle_dump(obj, path, raise_exceptions=True, verbose=True, **kwargs):
    """Wrapper for pickle.dump, accepting multiple path formats (file, string, pathlib.Path).
    - Automatically appends .p if not present.
    - Uses cpickle when possible.
    - Automatically closes file objects.

    Consider passing latest protocol:
    protocol=-1    OR
    protocol=pickle.HIGHEST_PROTOCOL"""
    if isinstance(path, Path):
        path = str(path)

    if isinstance(path, str):
        if path[-2:] != '.p':
            path += '.p'
        path = os.path.expanduser(path)
        with open(path, 'wb') as f:
            try:
                pickle.dump(obj, f, **kwargs)
            except TypeError as e:
                message = f'Failed to write pickle file: {path}. {e}'
                if raise_exceptions:
                    raise ValueError(message)
                else:
                    logger.warning(message)
                    success = False
            else:
                success = True
    else:
        try:
            pickle.dump(obj, path, **kwargs)
            path.close()
        except Exception as e:
            message = 'Filed to write pickle file. Unexpected path format/type: {}. {}'.format(path, e)
            if raise_exceptions:
                raise ValueError(message)
            else:
                logger.warning(message)
                success = False
        else:
            success = True

    if verbose:
        logger.info('Wrote pickle data to: %s' % path)

    return success


def pickle_load(path_fn, path=None, **kwargs):
    """Wrapper for pickle.load accepting multiple path formats (file, string, pathlib.Path).

    :param path_fn  : Filename or full path of pickle file
    :param path     : path in which path_fn is located (optional)
    :param kwargs   : keyword arguments to supply to pickle.load
    :return: Contents of pickle file
    """
    if isinstance(path_fn, Path):
        path_fn = str(path_fn)

    if path is not None:
        path_fn = os.path.join(path, path_fn)

    if isinstance(path_fn, str):
        if path_fn[-2:] != '.p':
            path_fn += '.p'

        try:
            with open(path_fn, 'rb') as f:
                out = pickle.load(f, **kwargs)
        except EOFError as e:
            logger.error('path "{}" is not a pickle file. {}'.format(path_fn, e))
            raise e
        except UnicodeDecodeError as e:
            try:
                kwargs.update({'encoding': 'latin1'})
                with open(path_fn, 'rb') as f:
                    out = pickle.load(f, **kwargs)
                logger.info('Reading pickle file required encoding="latin": {}'.format(path_fn))
            except Exception as e:
                logger.error('Failed to read pickle file "{}". Wrong pickle protocol? {}'.format(path_fn, e))
                raise e

    elif isinstance(path_fn, file):
        out = pickle.load(path_fn, **kwargs)
        path_fn.close()
    else:
        raise ValueError('Unexpected path format')
    return out


def json_dump(obj, path_fn, path=None, indent=4, overwrite=True, raise_on_fail=True):
    """Convenience wrapper for json.dump.

    Args:
        obj             : Object to be serialised
        path_fn         : Filename (and path) for output file
        path            : (Optional) path to output file
        indent          : Number of spaces for each json indentation
        overwrite       : Overwrite existing file
        raise_on_fail   : Whether to raise exceptions or return them

    Returns: Output file path if successful, else captured exception

    """
    import json
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
    import json
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
    out = utils.filter_nested_dict_key_paths(contents, key_paths_keep=key_paths_keep,
                                       compress_key_paths=compress_key_paths, path_fn=path_fn)

    # Drop some keys
    out = utils.drop_nested_dict_key_paths(out, key_paths_drop=key_paths_drop)

    if lists_to_arrays:
        out = utils.cast_lists_to_arrays(out)

    return out


def read_csv(path_fn: Union[Path, str], clean=True, convert_to_python_types=True, header='infer', comment='#',
             python_types_kwargs=None, raise_not_found=False, verbose=False, **kwargs):
    """Wrapper for pandas csv reader

    Other useful pandas.read_csv kwargs are:
    - names (names to use for column headings) if no header row

    Args:
        path_fn:                    Path to csv file
        clean:                      (bool) Remove erroneous column due to spaces at end of lines
        convert_to_python_types:    Convert eg "None" -> None, "true" -> True, "[1,2]" -> [1,2]
        header:                     Row number(s) to use as the column names, and the start of the data.
        comment:                    Comment charachter
        python_types_kwargs:        kwargs to pass to convert_dataframe_values_to_python_types:
                                    col_subset, allow_strings, list_delimiters, strip_chars
        raise_not_found:            Whether to raise to return exception if file not found
        verbose:                    Whether to log reading of file
        **kwargs: kwargs to pass to pd.read_csv, in particular:
                    'sep' column separator
                    'names' list of column names
                    'index_col' column name to set as index

    Returns: table pd.DataFrame of file contents

    """

    path_fn = Path(path_fn)
    if 'sep' not in kwargs:
        if path_fn.suffix == '.csv':
            kwargs['sep'] = '\s*,\s*'
        elif path_fn.suffix == '.tsv':
            kwargs['sep'] = r'\s+'
        elif path_fn.suffix == '.asc':
            kwargs['sep'] = r'\t'
        elif path_fn.suffix == '.txt':
            kwargs['sep'] = r'\s+'
        else:
            kwargs['sep'] = None  # Use csv.Sniffer tool
    try:
        with warnings.catch_warnings(record=True) as w:
            table = pd.read_csv(path_fn, header=header, comment=comment, **kwargs)  # index_col
    except FileNotFoundError as e:
        if raise_not_found:
            raise e
        else:
            return FileNotFoundError(f'CSV file does not exist: {path_fn}')
    except Exception as e:
        logger.warning(f'Failed to read file: {path_fn}')
        raise

    if clean:
        last_column = table.iloc[:, -1]
        try:
            all_nans = np.all(np.isnan(last_column))
        except TypeError as e:
            pass
        else:
            if all_nans:
                # TODO: Add keyword to make this optional?
                # Remove erroneous column due to spaces at end of lines
                table = table.iloc[:, :-1]
                logger.debug(f'Removed column of nans from csv file: {path_fn}')

    if convert_to_python_types:
        # Convert eg "None" -> None, "true" -> True, "[1,2]" -> [1,2]
        if python_types_kwargs is None:
            python_types_kwargs = {}
        table = convert_dataframe_values_to_python_types(table, **python_types_kwargs)

    if verbose:
        logger.info(f'Read data from file: {path_fn}')

    return table


def to_csv(path_fn: Union[Path, str], data, cols=None, index=None, x_range=None, drop_other_coords=False, sep=',',
           na_rep='nan', float_format='%0.5g', makedir=True, verbose=True,**kwargs):

    if makedir:
        mkdir(path_fn, verbose=True)

    if cols is not None:
        data = data[cols]

    if (index is not None) and (index not in data.dims):
        data = data.swap_dims({data.dims[0]: index})

    if x_range is not None:
        x = data.dims[0]
        data = data.sel({x: slice(*x_range)})

    if drop_other_coords:
        data = data.reset_coords()[cols]

    table = data.to_dataframe()

    table.to_csv(path_fn, sep=sep, na_rep=na_rep, float_format=float_format, **kwargs)

    if verbose:
        logger.info(f'Wrote data to file: {path_fn}')

    return table


def mkdir(dirs, start_dir=None, depth=None, accept_files=True, info=None, check_characters=True, verbose=1):
    """ Create a set of directories, provided they branch of from an existing starting directory. This helps prevent
    erroneous directory creation. Checks if each directory exists and makes it if necessary. Alternatively, if a depth
    is supplied only the last <depth> levels of directories will be created i.e. the path <depth> levels above must
    pre-exist.
    Inputs:
        dirs 			- Directory path
        start_dir       - Path from which all new directories must branch
        depth           - Maximum levels of directories what will be created for each path in <dirs>
        info            - String to write to DIR_INFO.txt file detailing purpose of directory etc
        check_characters- Whether to check for inappropriate characters in path, eg unformatted strings "/{pulse}/"
        verbose = 0	    - True:  print whether dir was created,
                          False: print nothing,
                          0:     print only if dir was created
    """
    from pathlib import Path
    # raise NotImplementedError('Broken!')
    if start_dir is not None:
        start_dir = os.path.expanduser(str(start_dir))
        if isinstance(start_dir, Path):
            start_dir = str(start_dir)
        start_dir = os.path.abspath(start_dir)
        if not os.path.isdir(start_dir):
            print('Directories {} were not created as start directory {} does not exist.'.format(dirs, start_dir))
            return 1

    if isinstance(dirs, Path):
        dirs = str(dirs)
    if isinstance(dirs, (str)):  # Nest single string in list for loop
        dirs = [dirs]
    # import pdb; pdb.set_trace()
    for d in dirs:
        if isinstance(d, Path):
            d = str(d)
        d = os.path.abspath(os.path.expanduser(d))
        if is_possible_filename(d):
            if accept_files:
                d = os.path.dirname(d)
            else:
                raise ValueError('mkdir was passed a file path, not a directory: {}'.format(d))
        if depth is not None:
            depth = np.abs(depth)
            d_up = d
            for i in np.arange(depth):  # walk up directory by given depth
                d_up = os.path.dirname(d_up)
            if not os.path.isdir(d_up):
                logger.info('Directory {} was not created as start directory {} (depth={}) does not exist.'.format(
                    d, d_up, depth))
                continue
        if not os.path.isdir(d):  # Only create if it doesn't already exist
            if (start_dir is not None) and (start_dir not in d):  # Check dir stems from start_dir
                if verbose > 0:
                    logger.info('Directory {} was not created as does not start at {} .'.format(dirs,
                                                                                          os.path.relpath(start_dir)))
                continue
            if check_characters and (not check_safe_path_string(d)):
                if not ask_input_yes_no(f'Path "{d}" contains invalid character. Do you still want to create it?'):
                    continue
            try:
                os.makedirs(d)
                if verbose > 0:
                    logger.info('Created directory: {}   ({})'.format(d, get_traceback_location(level=2)))
                if info:  # Write file describing purpose of directory etc
                    with open(os.path.join(d, 'DIR_INFO.txt'), 'w') as f:
                        f.write(info)
            except FileExistsError as e:
                logger.warning('Directory already created in parallel thread/process: {}'.format(e))
        else:
            if verbose > 1:
                logger.info('Directory "' + d + '" already exists')
    return 0


def split_path(path_fn, include_fn=False):
    if path_contains_fn(path_fn):
        path, fn = os.path.split(path_fn)
    else:
        path, fn = path_fn, None

    directories = list(Path(path).parts)
    if include_fn and (fn is not None):
        directories.append(fn)
    return directories, fn


def delete_file(fn, path=None, ignore_exceptions=(), raise_on_fail=True, verbose=True):
    """Delete file with error handelling
    :param fn: filename
    :param path: optional path to prepend to filename
    :ignore_exceptions: Tuple of exceptions to pass over (but log) if raised eg (FileNotFoundError,)
    :param raise_on_fail: Raise exception if fail to delete file
    :param verbose   : Print log messages
    """
    fn = str(fn)
    if path is not None:
        fn_path = os.path.join(path, fn)
    else:
        fn_path = fn
    success = False
    try:
        os.remove(fn_path)
        success = True
        if verbose:
            logger.info('Deleted file: {}'.format(fn_path))
    except ignore_exceptions as e:
        logger.debug(e)
    except Exception as e:
        if raise_on_fail:
            raise e
        else:
            logger.warning('Failed to delete file: {}'.format(fn_path))
    return success


def rm_files(paths, pattern, verbose=True, match=True, ignore_exceptions=()):
    """Delete files in paths matching patterns

    :param paths     : Paths in which to delete files
    :param pattern   : Regex pattern for files to delete
    :param verbose   : Print log messages
    :param match     : Use re.match instead of re.search (ie requries full pattern match)
    :param ignore_exceptions: Don't raise exceptions

    :return: None
    """
    paths = make_iterable(paths)
    for path in paths:
        path = str(path)
        if verbose:
            logger.info('Deleting files with pattern "{}" in path: {}'.format(pattern, path))
        for fn in os.listdir(path):
            if match:
                m = re.match(pattern, fn)
            else:
                m = re.search(pattern, fn)
            if m:
                delete_file(fn, path, ignore_exceptions=ignore_exceptions)
                if verbose:
                    logger.info('Deleted file: {}'.format(fn))


def check_safe_path_string(path):
    return bool(re.match('^[a-z0-9\.\/_]+$', path))


def is_possible_filename(fn, ext_whitelist=('py', 'txt', 'png', 'p', 'npz', 'csv',), ext_blacklist=(),
                         ext_max_length=3):
    """Return True if 'fn' is a valid filename else False.

    Return True if 'fn' is a valid filename (even if it and its parent directory do not exist)
    To return True, fn must contain a file extension that satisfies:
        - Not in blacklist of extensions
        - May be in whitelist of extensions
        - Else has an extension with length <= ext_max_length
    """
    fn = str(fn)
    ext_whitelist = ['.' + ext for ext in ext_whitelist]
    if os.path.isfile(fn):
        return True
    elif os.path.isdir(fn):
        return False

    ext = os.path.splitext(fn)[1]
    l_ext = len(ext) - 1
    if ext in ext_whitelist:
        return True
    elif ext in ext_blacklist:
        return False

    if (l_ext > 0) and (l_ext <= ext_max_length):
        return True
    else:
        return False


def sub_dirs(path):
    """Return subdirectories contained within top level directory/path"""
    path = os.path.expanduser(path)
    assert os.path.isdir(path)
    out = [p[0] for p in os.walk(path)]
    if len(out) > 0:
        out.pop(out.index(path))
    return out


def path_contains_fn(path_fn):
    _, ext = os.path.splitext(path_fn)
    if ext == '':
        out = False
    else:
        out = True
    return out


def copy_file(src, dest, extensions_text=('txt', 'csv', 'tsv'), text=None, mkdir_dest=False, verbose=True,
              raise_exceptions=True):
    src = Path(src)
    dest = Path(dest)

    if mkdir_dest:
        mkdir(dest, verbose=verbose)

    try:
        if text is True or src.suffix in extensions_text:
            dest.write_text(src.read_text())  # for text files
        else:
            dest.write_bytes(src.read_bytes())  # for binary files
    except Exception as e:
        if raise_exceptions:
            raise e
        else:
            if verbose:
                logger.info(f'Failed to copy "{src}" to "{dest}": {e}')
    else:
        if verbose:
            logger.info(f'Copied "{src}" to "{dest}"')

if __name__ == '__main__':
    pass
