import configparser
import os, time, gc, logging, re, shutil
from pathlib import Path
from datetime import datetime
import collections

import numpy as np
import pandas as pd

from fire.interfaces.io_basic import mkdir, split_path
from fire.misc.utils import (make_iterable, is_subset, str_to_number, args_for,
                             print_progress, argsort, mkdir)

logger = logging.getLogger(__name__)
try:
    from natsort import natsorted
    sorted = natsorted
except ImportError as e:
    logger.debug('Please install natsort for improved sorting')

try:
    from past.types import basestring
    string_types = (basestring, unicode)  # python2
except Exception as e:
    string_types = (str,)  # python3


def insert_subdir_in_path(path_fn, subdir, position=-1, keep_fn=True, create_dir=True):
    """Insert a subdirectory in a path"""
    parts, fn = split_path(path_fn, include_fn=False)
    if position < 0:
        position = len(parts) + 1 + position
    parts.insert(position, subdir)
    new_path = '/'.join(parts)
    if new_path[:2] == '//':
        new_path = new_path[1:]
    if not os.path.isdir(new_path) and create_dir:
        depth = abs(position) if position < 0 else (len(parts)+2 - position)
        mkdir(new_path, depth=depth)
    if keep_fn and (fn is not None):
        new_path = os.path.join(new_path, fn)
    return new_path

def create_config_file(fn, dic):
    """Create ini config file structured around supplied dictionary"""
    config = configparser.ConfigParser()
    for key, value in dic.items():
        if isinstance(value, dict):
            config[key] = {}
            for k, v in value.items():
                config[key][k] = v
        else:
            config[key] = value
    with open(fn, 'w') as configfile:
        config.write(configfile)
    logging.info('Wrote config file to {}. Sections: {}'.format(fn, config.sections()))


def getUserFile(type=""):
    from tkinter import Tk, filedialog as askopenfilename
    Tk().withdraw()
    filename = askopenfilename(message="Please select "+type+" file:")
    return filename

def filter_files_in_dir(path, fn_pattern='.*', group_keys=(), modified_range=(None, None), n_matches_expected=None,
                        return_full_paths=False, modified_dir_filter=False, raise_on_incomplete_match=False,
                        raise_on_missing_dir=True, raise_on_no_matches=True, depth=0, include_empty_dirs=False,
                        sort_keys=True, **kwargs):
    """Return dict of filenames in 'path' directory that match supplied regex pattern

    The keys of the returned dict are the matched groups for each file from the fn_pattern.
    :param filenames: filenames to be filtered
    :param fn_pattern: regex pattern to match against files. kwargs will be substituted into the pattern (see example)
    :param path: path where files are located (only needed to querying files modification dates etc)
    :param group_keys: list that links the ordering of the regex groups to the kwargs keys. Warnings are raised
                         if not all of the kwarg values are matched to files.
    :param modified_range: Age range in days to accept [n_days_old_min, n_days_old_max]
                            eg modified_range=[None, 3] filters all files modified in the last 3 days
    :param return_full_paths: Prepend paths to output file names
    :param raise_on_incomplete_match: raise an exception if not all kwarg values are located
    :param kwargs: values are substituted into the fn_pattern (provided the pattern contains a format key matching that
                    of the kwarg) with lists/arrays of values converted to the appropriate regex pattern.
    e.g. to get the files with values of n between 10 and 50 use. The returned dict will be keyed by the number n and
    the last group in the filename (<n>, <.*>). You will be warned if any of the requested n's are not found.
    fn_pattern = 'myfile-n({n})-(.*).nc'
    fns = filter_files_in_dir(path, fn_pattern, group_keys=['n'], n=np.arange(20,51))

    """
    path = Path(path).expanduser()
    if not path.is_dir():
        if raise_on_missing_dir:
            raise IOError('Search directory "{}" does not exist'.format(path))
        else:
            return {}
    path = path.resolve()
    # filenames_all = sorted(os.listdir(str(path)))
    out = {}
    n_matches = 0
    # Loop over paths
    for i, (root, dirs, files) in enumerate(os.walk(str(path), topdown=True)):
        level = root.replace(str(path), '').count('/')
        if level > depth:
            break
        if modified_dir_filter:
            raise NotImplementedError
        # TODO: Raise separate error for no matches with depth>0
        raise_on_no_matches = raise_on_no_matches and (depth == 0)
        out_i = filter_files(files, fn_pattern, path=root, group_keys=group_keys, n_matches_expected=n_matches_expected,
                             modified_range=modified_range, raise_on_incomplete_match=raise_on_incomplete_match,
                             raise_on_no_matches=raise_on_no_matches, verbose=False, **kwargs)
        if (len(out_i) == 0) and not include_empty_dirs:
            continue
        else:
            n_matches += len(out_i)
            out[root] = out_i
        if (n_matches_expected is not None) and (n_matches >= n_matches_expected):
            # If all the required files have been found, ignore the remaining files
            break
    if (n_matches == 0):
        message = 'Failed to locate any files with pattern "{}" in {}, depth={}'.format(fn_pattern, path, depth)
        if raise_on_no_matches:
            raise IOError(message)
        else:
            logger.warning(message)
    if (depth == 0) and (len(out) == 1):
        # If depth is 0, don't nest in directory dict
        path = list(out.keys())[0]
        out = out[path]
        if return_full_paths:
            # Prepend paths to fileneames
            out = {key: os.path.join(path, fn) for key, fn in out.items()}
    elif return_full_paths:
        # Return single level dict of key: fn pairs with abs paths
        out_tmp = out
        out = {}
        n_files = 0
        for path, fns in out_tmp.items():
            path = os.path.abspath(os.path.expanduser(path))
            tmp = {key: os.path.join(path, fn) for key, fn in fns.items()}
            out.update(tmp)
            n_files += len(tmp)
            assert len(out) == n_files, ('filter_files_in_dir returned multiple files with same file key: '
                                         '{}/{} files in {}'.format(n_files-len(out), len(tmp), tmp))
        assert len(out) == n_matches

    if sort_keys:
        keys, fns = list(out.keys()), list(out.values())  # improve
        i_sort = argsort(keys)
        ordered_keys = np.array(keys)[i_sort]
        if ordered_keys.ndim > 1:
            # ndarray is not hashable
            ordered_keys = [tuple(v) for v in ordered_keys]
        out = collections.OrderedDict(zip(ordered_keys, np.array(fns)[i_sort]))
        # out = {key: [[os.path.join(ps, f) for f in fns] for ps, fns in out_tmp.items()]}
        # out = itertools.chain(*out)
    return out


def filter_files(filenames, fn_pattern, path=None, group_keys=(), modified_range=(None, None), n_matches_expected=None,
                 raise_on_incomplete_match=False, raise_on_no_matches=True, verbose=True, **kwargs):
    """Return dict of filenames from given set of filenames that match supplied regex pattern

    The keys of the returned dict are the matched groups for each file from the fn_pattern.
    :param filenames: filenames to be filtered
    :param fn_pattern: regex pattern to match against files. kwargs will be substituted into the pattern (see example)
    :param path: path where files are located (only needed to querying files modification dates etc)
    :param group_keys: list that links the ordering of the regex groups to the kwargs keys. Warnings are raised
                         if not all of the kwarg values are mmatched to files.
    :param modified_range: Age range in days to accept [n_days_old_min, n_days_old_max]
                            eg modified_range=[None, 3] filters all files modified in the last 3 days
    :param raise_on_incomplete_match: raise an exception if not all kwarg values are located
    :param kwargs: values are substituted into the fn_pattern (provided the pattern contains a format key matching that
                    of the kwarg) with lists/arrays of values converted to the appropriate regex pattern.

    e.g. to get the files with values of n between 10 and 50 use. The returned dict will be keyed by the number n and
    the last group in the filename (<n>, <.*>). You will be warned if any of the requested n's are not found.
    fn_pattern = 'myfile-n({n})-(.*)\.nc'
    fns = filter_files_in_dir(path, fn_pattern, group_keys=['n'], n=np.arange(20,51))

    e.g. to get all files in a directory sorted by some number in their filename i.e. catch all values of n:
    fn_pattern = 'myfile-n(\d+)-.*\.nc'
    fns = filter_files_in_dir(path, fn_pattern, group_keys=['n'])
    """
    # TODO: Use glob for path selection
    # from ccfepyutils.utils import PartialFormatter
    # fmt = PartialFormatter()

    if (modified_range != (None, None)):
        assert path is not None, 'A path must be supplied to filter files by modified date'
        assert len(modified_range) == 2, 'Modifed range must have start and end'
        assert os.path.isdir(path)

    n_files = len(filenames)

    # If kwargs are supplied convert them to re patterns
    re_patterns = {}
    for key, value in kwargs.items():
        if isinstance(value, (np.ndarray, list, tuple)):  # and is_number(value[0]):
            # List/array of numbers, so match any number in list
            re_patterns[key] = '{}'.format('|'.join([str(v) for v in value]))
        elif isinstance(value, str):
            # Replace python format codes eg {time:0.3f} with regex pattern eg ([.\d]{3,4})
            fmt_pattern = r'{{{key:}[^_]*}}'.format(key=key).replace(r'{', r'\{').replace(r'}', r'\}')
            fn_pattern = re.sub(fmt_pattern, value, fn_pattern)
    # fn_pattern = fn_pattern.format(**re_patterns)
    try:
        # fn_pattern = fmt.format(fn_pattern, **re_patterns)
        fn_pattern = fn_pattern.format(**re_patterns)
    except IndexError as e:
        pass
    # Allow custom \e+ escape sequence to capture standard form numbers
    # optional sign,
    # .1 .12 .123 etc 9.1 etc 98.1 etc or # 1. 12. 123. etc 1 12 123 etc
    #  followed by optional exponent part if desired
    numeric_const_pattern = '[-+]? (?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )(?: [Ee] [+-]? \d+ ) ?'
    fn_pattern = fn_pattern.replace('\e+', numeric_const_pattern)
    out = {}
    i = 0
    t0 = datetime.now()
    for j, fn in enumerate(filenames):
        # Check if each filename matches the pattern
        if n_files > 500:
            print_progress(j, n_files, t0=t0, nth_loop=50, prefix='Filtering files')
        rx = re.compile(fn_pattern, re.VERBOSE)
        m = rx.search(fn)
        if m is None:
            continue
        if path is not None:
            fn_path = os.path.join(path, fn)
            t_now = time.time()
            t_day = 24*60*60
            t_age = t_now-os.path.getmtime(fn_path)
            t_age /= t_day
            # Modified_range = [n_days_old_min, n_days_old_max]
            if (modified_range[0] is not None) and (t_age < modified_range[0]):
                continue
            if (modified_range[1] is not None) and (t_age > modified_range[1]):
                continue

        ngroups = len(m.groups())
        if ngroups == 0:
            # Use index of element as output key
            key = i
        elif ngroups == 1:
            # Remove nesting tuple
            key = str_to_number(m.groups()[0])
        else:
            # Use tuple of elements from pattern matches as key
            key = tuple(str_to_number(v) for v in m.groups())
        out[key] = fn
        i += 1
        if (n_matches_expected is not None) and (i == n_matches_expected):
            logger.debug('Located expected number of files: {}'.format(i))
            break

    if len(out) == 0:
        message = 'Failed to locate any files with pattern "{}" in {}'.format(fn_pattern, filenames)
        if raise_on_no_matches:
            raise IOError(message)
        else:
            if verbose:
                logger.warning(message)
            return {}
    for i, group_key in enumerate(group_keys):
        if (group_key not in kwargs) or (isinstance(kwargs[group_key], (str, type(None)))):
            continue
        # List of located values for group cast to same type
        if ngroups == 0:
            raise ValueError('fn_pattern doesnt not contain any regex groups "()"')
        if ngroups == 1:
            located_values = list(out.keys())
        else:
            located_values = [type(kwargs[group_key][0])(key[i]) for key in out.keys()]
        if not is_subset(kwargs[group_key], list(located_values)):
            message = 'Could not locate files with {} = {}'.format(group_key,
                                                                  set(kwargs[group_key]) - set(located_values))
            if raise_on_incomplete_match:
                raise RuntimeError(message)
            else:
                if verbose:
                    logger.warning(message)
    return out

def age_of_file(fn_path):
    t_now = time.time()
    t_day = 24*60*60
    t_age = t_now-os.path.getmtime(fn_path)
    return t_age


def delete_files_recrusive(pattern, path=None, delete_files=True, delete_directories=False, prompt_user=True):
    from fire.misc.utils import ask_input_yes_no
    if path is None:
        path = '.'
    path = Path(path).resolve()
    h = re.compile(pattern)
    to_be_removed = {'files': [], 'dirs': []}
    for root, dirs, files in os.walk(path):
        if delete_directories:
            for dir0 in filter(lambda x: h.match(x), dirs):
                path_fn = os.path.join(root, dir0)
                to_be_removed['dirs'].append(path_fn)
                if not prompt_user:
                    os.remove(path_fn)
        if delete_files:
            for file in filter(lambda x: h.match(x), files):
                path_fn = os.path.join(root, file)
                to_be_removed['files'].append(path_fn)
                if not prompt_user:
                    os.remove(path_fn)
    n_files = len(to_be_removed["files"])
    n_dirs = len(to_be_removed["dirs"])

    if prompt_user:
        print(f'{n_files} files and {n_dirs} directories to be deleted: \n{to_be_removed}'.format(
            to_be_removed=to_be_removed))
        if ask_input_yes_no('Delete files'):
            for file in to_be_removed['files']:
                os.remove(file)
            for directory in to_be_removed['dirs']:
                shutil.rmtree(directory)
            print(f'{n_files} files and {n_dirs} directories were deleted')
        else:
            print('No files were deleted')
    else:
        print(f'The following {n_files} files and {n_dirs} directories were deleted: \n{to_be_removed}'.format(
            to_be_removed=to_be_removed))


def arrays_to_csv(x, ys, fn, xheading=None, yheadings=None, description='data'):
    """Quickly and easily save data to csv, with one dependent variable"""
    import pandas as pd
    ys = np.squeeze(ys)
    if ys.shape[0] != len(x) and ys.shape[1] == len(x):
        ys = ys.T
    df = pd.DataFrame(data=ys, index=make_iterable(x), columns=yheadings)
    if xheading:
        df.index.name = xheading
    # df.columns.name = 'tor'
    df.to_csv(str(fn))
    logger.info('Saved {} to: {}'.format(description, str(fn)))


def out_path(input, default_path, default_fn, default_extension, path_obj=False):
    """Generate output path given input and variable input.

    Any information missing in input with be replaced with defaults.
    Input can be eg:
    Full path including filename and extension
    Just path, no filename
    Just filename, no path
    Not a string/Path, in which case defaults will be used """
    default_path = os.path.expanduser(default_path)
    try:
        if isinstance(input, Path):
            input = str(input)
        input = os.path.expanduser(input)
        if os.path.isdir(input) and input[-1] != os.sep:  # Make sure a directory ends in slash
            input += os.sep
        # Split up input information
        path, fn = os.path.split(input)
        base, ext = os.path.splitext(fn)
        if len(ext) > 5:  # Avoid full stops in filenames being treated as extensions - 4 char ext len limit
            base += ext
            ext = ''

        # Replace missing information with defaults
        if not path:
            path = default_path
        if not fn:
            fn = os.path.splitext(default_fn)[0]
        if not ext:
            ext = default_extension
        fn = join_with_one('.', fn, ext)
        out = os.path.join(path, fn)

    except AttributeError as e:
        # TODO: allow no extension and extension included in default_fn
        fn = join_with_one('.', default_fn, default_extension)
        out = os.path.join(default_path, fn)
    if path_obj:
        out = Path(out)
    return out

def join_with_one(sep, *args):
    """Join strings with exactly one separator"""
    l = len(sep)
    out = ''
    for i, (arg1, arg2) in enumerate(zip(args, args[1:])):
        arg12 = [arg1, arg2]
        if i == 0:
            # If arg1 ends with separator, remove it
            while arg12[0].endswith(sep):
                arg12[0] = arg12[0][:len(arg12[0])-l]
            out += arg12[0]
        # check if arg2 begins or ends with separator (need to loop over mutable!)
        while arg12[1].startswith(sep):
            arg12[1] = arg12[1][l:]
        while arg12[1].endswith(sep):
            arg12[1] = arg12[1][:len(arg12[1])-l]
        # add to output
        out += sep+arg12[1]
    return out

def python_path(filter=None):
    import os
    try:
        user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
    except KeyError:
        print('PYTHONPATH not set')
        user_paths = []
    if filter:
        filter = os.path.expanduser(filter)
        user_paths = [p for p in user_paths if filter in p]
    return user_paths

def locate_file(paths, fns, path_kws=None, fn_kws=None, return_raw_path=False, return_raw_fn=False, _raise=True,
                verbose=False):
    """Return path to file given number of possible paths"""
    # TODO: detect multiple occurences/possible paths
    if path_kws is None:
        path_kws = {}
    if fn_kws is None:
        fn_kws = {}

    for path_raw in paths:
        # Insert missing info in
        path_raw = str(path_raw)
        path = path_raw.format(**path_kws)
        try:
            path = Path(path).expanduser()
        except RuntimeError as e:
            if "Can't determine home directory" in str(e):
                continue
            else:
                raise e
        if not path.is_dir():
            continue
        path = path.resolve()
        for fn_raw in fns:
            fn = str(fn_raw).format(**fn_kws)
            fn_path = path / fn
            if fn_path.is_file():
                path_out = path_raw if return_raw_path else path
                fn_out = fn_raw if return_raw_fn else fn
                if verbose >= 2:
                    logging.info('Located "{}" in {}'.format(fn_out, path_out))
                return path_out, fn_out
    if _raise:
        raise IOError('Failed to locate file in paths "{}" with formats: \n{}'.format(list(paths), list(fns)))
    else:
        if verbose:
            logger.warning('Failed to locate file in paths "{}" with formats: {}'.format(paths, fns))
        return None, None

class AttemptError(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)

def attempt_n_times(func, args=None, kwargs=None, n_attempts=3, sleep_interval=0.5,
                    exceptions=(IOError,), requried_output=None,
                    error_message='Call to {func} failed after {n_attempts} attempts',
                    call_on_fail=(), raise_on_fail=True, verbose=True):
    """Attempt I/O call multiple times with pauses in between to avoid read/write clashes etc."""
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    if requried_output is not None:
        requried_output = make_iterable(requried_output)
    args = make_iterable(args)
    call_on_fail = make_iterable(call_on_fail)
    attempt = 1
    success = False
    while (success is False):
        try:
            # logger.debug('Attempt {} to call function "{}({})"'.format(
            #                 attempt, func.__name__, ', '.join([str(a) for a in args])))
            out = func(*args, **kwargs)
            if (requried_output is not None) and (out not in requried_output):
                raise AttemptError('Function output "{}" not in required values: {}'.format(out, requried_output))
            success = True
            # logger.debug('Suceeded on attempt {} to call function "{}({})"'.format(
            #                 attempt, func.__name__, ', '.join([str(a) for a in args])))
        except (exceptions + (AttemptError,)) as e:
            logger.warning('Attempt {} to call function "{}({}, {})" failed'.format(
                            attempt, func.__name__, ', '.join([str(a) for a in args]),
                            ', '.join(['{}={}'.format(k, v) for k, v in kwargs.items()])))
            gc.collect()  # Attemp to avoid netcdf segfaults?
            if attempt <= n_attempts:
                time.sleep(sleep_interval)
                attempt += 1
            else:
                if error_message is not None:
                    try:
                        logger.error(error_message.format(func=func.__name__, n_attempts=n_attempts))
                    except Exception as e:
                        raise e
                for func in call_on_fail:
                    raise NotImplementedError  # Need args_for to pass positional args
                    args, kwargs = args_for(func, kwargs)
                    func(*args0, **kws)
                if raise_on_fail:
                    raise e
                else:
                    out = e
                    break
        except Exception as e:
            raise e
    return out, success

def gen_hash_id(obj, algorithm='ripemd160'):
    """Generate a unique hash string for the supplied object.

    The string representation of the object, encoded into bytes, is used to generate the hash.
    Previously the default hashing algorithm was ripemd160"""
    import hashlib
    h = hashlib.new(algorithm)
    h.update(bytes(str(obj), 'utf-8'))
    hash_id = h.hexdigest()
    return hash_id


class HashId:

    # def __new__(cls, obj, algorithm='md5', name=None, object_description=None, meta_data=None):
    #     instance = str.__new__(cls, '')
    #     return instance

    def __init__(self, obj, algorithm='md5', name=None, object_description=None, meta_data=None):
        self.name = name
        self.object_description = object_description
        self.hash_obj = obj
        self.hash_meta = meta_data
        self.algorithm = algorithm
        self.hash_id = gen_hash_id(obj, algorithm)
        # self += self.hash_id

    def __repr__(self):
        return '<HashId({}):{}>'.format(self.name, self.hash_id)

    def __str__(self):
        return '{}'.format(self.hash_id)

    def __eq__(self, other):
        out = False
        if isinstance(other, str):
            if other == self.hash_id:
                out = True
            elif HashId(other).hash_id == self.hash_id:
                out = True
        elif isinstance(HashId) and (other.hash_id == self.hash_id):
            out = True
        else:
            if HashId(other).hash_id == self.hash_id:
                out = True
        return out

    def info(self):
        info = pd.Series({'hash_id': self.hash_id, 'hash_meta': self.hash_meta, 'algorithm': self.algorithm,
                          'name': self.name, 'object_description': self.object_description})
        return info


def fn_filter(path, fn_pattern, recursive=False, unique=False):
    """ Filenames in a given directory that match the search pattern
    TODO: add compatibility for non raw string file paths
    """
    fns = os.listdir(path)
    p = re.compile(fn_pattern)
    matches = []
    for fn in fns:
        if p.search(fn):
            matches.append(fn)
    if matches == []:
        print('No files match the supplied pattern: "%s"' % fn_pattern)
    if unique:  # expect unique match so return scalar string that matches
        if len(matches) == 1:
            return matches[0]
        else:
            raise ValueError('WARNING: fn_filter(unique=True): {} matches: {}'.format(len(matches), matches))
    else:
        return matches

def fn_filter_numeric_range(path_in, fn_pattern, numeric_range, sort_output=True):
    """Return sorted subset of filenames within a directory, within a numeric range

    The fn_pattern must contain {number} which must contain an integer in the numeric range"""
    assert '{number}' in fn_pattern, 'Include "{number}" in pattern when using file range'
    fn_pattern = fn_pattern.format(number=regexp_int_range(*numeric_range))

    filenames = fn_filter(path_in, fn_pattern)
    if sort_output:
        filenames = sorted(filenames)
    return filenames

def extract_fn_path(path_fn, path=None):
    """Return separated path and filename for given inputs
    path_fn could be a full path and filename or just a filename
    path could be a filepath or None
    """
    if os.path.split(path_fn)[0] == '':
        fn = path_fn
        if (path is None):
            path = './'
    else:
        # TODO: Consider appending path_fn to path? Check path_fn is relative?
        assert (path is None), 'Path "{}" supplied when path_fn already contains path info: {}'.format(path, path_fn)
        path = os.path.dirname(path_fn)
        fn = os.path.basename(path_fn)

    fn, ext = os.path.splitext(fn)
    return path, fn, ext

def combine_fn_path(path_fn, path=None):
    """Return Path object given full file path or filename and path separately"""
    path, fn, ext = extract_fn_path(path_fn, path=path)
    path_fn = Path(path) / (fn+ext)
    return path_fn

def format_extension(extension, with_dot=False, none_for_empty=False):
    """Add or remove leading '.' to file extension string"""
    assert isinstance(extension, str)
    if (extension == ''):
        if none_for_empty:
            extension = None
        return extension

    if with_dot:
        if extension[0] != '.':
            extension = '.' + extension
    else:
        if extension[0] == '.':
            extension = extension[1:]
    return extension

def check_path_characters_are_safe(path_fn, allow_home_tilde=True, raise_error=True):
    """"Check path only contains safe characters: A-Z a-z 0-9 _ - / ."""
    assert isinstance(path_fn, (str, Path))
    path_fn = str(path_fn)
    assert len(path_fn) > 0

    if allow_home_tilde and (path_fn[0]) == '~':
        path_fn = path_fn[1:]

    m = re.findall(r'[^A-Za-z0-9_\-/\.]', path_fn)
    if m:
        if raise_error:
            raise ValueError('Path_fn "{}" contains invalid/unsafe characters: {}'.format(path_fn, m))
        else:
            out = False
    else:
        out = True
    return out

def regexp_int_range(low, high, compile=False):
    fmt = '%%0%dd' % len(str(high))
    if compile:
        return re.compile('(%s)' % '|'.join(fmt % i for i in range(low, high + 1)))
    else:
        return '(%s)' % '|'.join('{:d}'.format(i) for i in range(low, high + 1))

def regexp_int_set(values, compile=False):
    fmt = '%%0%dd' % len(str(np.max(values)))
    if compile:
        return re.compile('(%s)' % '|'.join(fmt % i for i in values))
    else:
        return '(%s)' % '|'.join('{:d}'.format(i) for i in values)

def pos_path(value, allow_relative=True):
    """Return True if value is a potential file path else False"""
    if not isinstance(value, string_types):
        return False
    value = os.path.expanduser(value)
    if allow_relative:
        value = os.path.abspath(value)
    path, fn = os.path.split(value)
    if os.path.isdir(path):
        return True
    # elif (allow_relative) and (path == ''):
    #     return True
    else:
        return False

def read_netcdf_group(fn_path, group):
    import xarray as xr
    with xr.open_dataset(fn_path, group=group, autoclose=True) as match_data:
            match_data = match_data.copy(deep=True)
    return match_data

def merge_pdfs(fns_in, fn_out):
    """Merge list of pdf (portable document format) files into a single document"""
    from PyPDF2 import PdfFileReader, PdfFileWriter
    pdf_writer = PdfFileWriter()

    fns_in = make_iterable(fns_in)
    for path in fns_in:
        pdf_reader = PdfFileReader(path)
        for page in range(pdf_reader.getNumPages()):
            # Add each page to the writer object
            pdf_writer.addPage(pdf_reader.getPage(page))

    # Write out the merged PDF
    with open(fn_out, 'wb') as out:
        pdf_writer.write(out)
    logger.info('Combined {l} pdf documents into file: {fn_out}'.format(l=len(input_fns), fn_out=fn_out))

def extract_pdf_pages(input_fn, fn_out_pattern='{input_stem}_p{page_min}-{page_max}.pdf', pages='all'):
    """Extract pages from pdf (portable document format) file to a new file.

    Page numbering starts at 1"""
    from PyPDF2 import PdfFileReader, PdfFileWriter
    pages = make_iterable(pages)
    input_stem = Path(input_fn).resolve().stem
    pdf_out_writer = PdfFileWriter()
    pdf = PdfFileReader(input_fn)
    for page_no in (np.arange(pdf.getNumPages())+1):
        if (page_no in pages) or (pages == ['all']):
            print('extracting page {page_no}'.format(page_no=page_no))
            pdf_out_writer.addPage(pdf.getPage(page_no))
        else:
            print('Skipping page {page_no}'.format(page_no=page_no))

    fn_out = fn_out_pattern.format(input_stem=input_stem, page_min=min(pages), page_max=max(pages))
    with open(fn_out, 'wb') as output_pdf:
        pdf_out_writer.write(output_pdf)
    logger.info('Wrote pages {pages} of pdf {input_fn} to: {fn_out}'.format(pages=pages, input_fn=input_fn, fn_out=fn_out))




if __name__ == '__main__':
    # path = '/home/tfarley/elzar2/checkpoints/MAST/SynthCam/single_filament_scan/Corrected_inversion_data/6bb2ed99e9772ce84f1fba74faf65e23a7e5e8f3/'
    # fn_pattern = 'corr_inv-test1-n({n})-6bb2ed99e9772ce84f1fba74faf65e23a7e5e8f3.nc'
    # fns = filter_files_in_dir(path, fn_pattern, group_keys=['n'], n=np.arange(4600,4650), depth=1)
    path = r'/home/tfarley/elzar2/checkpoints/MAST/SA1.1/29852/Detected_blobs/'
    # fn_pattern = 'corr_inv-test1-n({n})-\w+.nc'
    fn_pattern = r'blobs-test1-n({n})-\w+.nc'
    # Get files modifed in the last two days
    fns = filter_files_in_dir(path, fn_pattern, group_keys=['n'], n=np.arange(10500, 14500), depth=1, modified_range=(0, 2))

    fn = os.path.expanduser(r'~/repos/elzar2/elzar2/default_settings/elzar_defaults.ini')
    # from nested_dict import nested_dict
    # file = nested_dict()
    # file['Paths']['elzar_path'] = '~/elzar/:'
    # file['Paths']['data'] = ''
    #
    # file['Movie']['source'] = 'repeat'
    #
    # file['Invertor']['type'] = 'PsfInvertor'
    # file['Invertor']['settings'] = 'repeat'
    # file['Invertor']['resolution'] = 'repeat'
    #
    # file['Detector']['type'] = 'QuadMinEllipseDetector'
    # file['Detector']['settings'] = 'repeat'
    #
    # file['Tracker']['type'] = 'NormedVariationTracker'
    # file['Tracker']['settings'] = 'repeat'
    #
    # file['Benchmarker']['type'] = 'ProximityBenchmarker'
    # file['Tracker']['settings'] = 'repeat'
    # # file['elzar_path']['path'] = os.path.expanduser('~/elzar/')
    #


