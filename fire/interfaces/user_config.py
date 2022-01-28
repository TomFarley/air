#!/usr/bin/env python

"""


Created: 
"""

import logging, os
from typing import Union, Iterable, Sequence, Tuple, Optional, Any, Dict
from pathlib import Path
from copy import copy
from functools import lru_cache
from collections import namedtuple

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# To enable use in fire/__init__.py avoid any imports that import fire directly
import fire
from fire import PATH_AIR_REPO, PATH_FIRE_SOURCE
from fire.interfaces import io_basic
# from fire.misc import utils
# from fire.misc.utils import make_iterable

logger = logging.getLogger(__name__)

_FIRE_CONFIG_TEMPLATE_PATH_FN = PATH_FIRE_SOURCE / 'input_files' / 'user' / 'fire_config.json'
_FIRE_USER_DIR_ENV_VAR = 'FIRE_USER_DIR'
_FIRE_USER_DIR_DEFAULT = "~/fire/"
_FIRE_USER_CONFIG_FN_DEFAULT = 'fire_config.json'

_FIRE_USER_DIR_KEY = 'fire_user_dir'

@lru_cache(maxsize=8, typed=False)
def get_fire_user_directory(fire_user_dir=None):

    if (fire_user_dir is not None) and (Path(fire_user_dir).expanduser().is_dir()):
        if Path(fire_user_dir) != fire.user_path:
            logger.info(f'Using user supplied FIRE user directory: "{fire_user_dir}"')
    else:
        fire_user_dir = os.environ.get(_FIRE_USER_DIR_ENV_VAR, None)
        if (fire_user_dir is None):
            if fire.user_path is not None:
                fire_user_dir = fire.user_path
            else:
                fire_user_dir = _FIRE_USER_DIR_DEFAULT
                print(f'- Environment variable "{_FIRE_USER_DIR_ENV_VAR}" has not been set. Using default path '
                      f'"{_FIRE_USER_DIR_DEFAULT}" for configuration file and user outputs.\n'
                      f'- To set your "{_FIRE_USER_DIR_ENV_VAR}" environment variable (on linux) do:\n'
                      f'- $ export {_FIRE_USER_DIR_ENV_VAR}="<my_path>"  # It is recommended to add this to your .bashrc')
                os.environ[_FIRE_USER_DIR_ENV_VAR] = str(fire_user_dir)  # visible in this process + all children
        else:
            if Path(fire_user_dir) != fire.user_path:
                logger.info(f'Using FIRE user directory "{fire_user_dir}" from env var '
                         f'{_FIRE_USER_DIR_ENV_VAR}="{fire_user_dir}"')
    fire_user_dir = Path(fire_user_dir)

    if not fire_user_dir.is_dir():
        fire_user_dir.mkdir()
        logger.info(f'Created FIRE user directory at: {fire_user_dir}')

    fire.user_path = Path(fire_user_dir)

    return fire_user_dir

@lru_cache(maxsize=2)
def get_fire_user_config_path_fn(path=None, fn=None):
    if fn is None:
        fn = _FIRE_USER_CONFIG_FN_DEFAULT

    path = get_fire_user_directory(fire_user_dir=path)

    if not path.is_file():
        path_fn = path / fn
    else:
        path_fn = path

    return path_fn

def _read_user_fire_config(path=None, fn=None):
    """Read users FIRE configuration file (eg '~/.fire/.fire_config.json')

    Args:
        path: File path to directory containing config file
        fn:   Filename of config file

    Returns: Contents of config file

    """
    path_fn = get_fire_user_config_path_fn(path, fn)

    fire_config = io_basic.json_load(path_fn, key_paths_drop=('README',))

    return fire_config

def update_base_paths_in_fire_config(fire_config, base_paths_user=None, update_inplace=True):
    fire_user_path = get_fire_user_directory()

    base_paths = {_FIRE_USER_DIR_KEY: fire_user_path, 'air_repo_dir': PATH_AIR_REPO, 'fire_source_dir': PATH_FIRE_SOURCE}
    base_paths.update(base_paths_user)

    path_aliases = fire_config['user']['path_aliases']

    paths, base_paths = insert_base_paths('paths', fire_config['user']['paths'], base_paths, path_aliases=path_aliases,
                              update_inplace=update_inplace)
    if update_inplace:
        fire_config['user']['paths'] = paths

    return base_paths, fire_config

def insert_base_paths(key, path, base_paths, path_aliases=(), update_inplace=True):
    path_aliases = dict(path_aliases)

    path_out = path if update_inplace else copy(path)

    if isinstance(path, (dict)):
        for key, paths in path.items():
            path_updated, base_paths = insert_base_paths(key, paths, base_paths, path_aliases=path_aliases,
                                                      update_inplace=update_inplace)
            path_out[key] = path_updated

    elif isinstance(path, (list)):
        for i, path_i in enumerate(path):
            path_updated, base_paths = insert_base_paths(key, path_i, base_paths, path_aliases=path_aliases,
                                                      update_inplace=update_inplace)
            path_out[i] = path_updated

    elif isinstance(path, str):
        path_out = Path(path.format(**base_paths))
        key_alias = path_aliases.get(key, key)

        if (key_alias not in base_paths) or (base_paths.get(key_alias) is None):  # take first value when there are
            # multiple options
            base_paths[key_alias] = path_out

    else:
        raise ValueError(path)

    return path_out, base_paths

def copy_default_user_settings(path=None, fn=None, replace_existing=False):
    # Set up fire config file

    path_fn = get_fire_user_config_path_fn(path, fn)

    # TODO: skip if on scheduler?
    if (not path_fn.is_file()) or replace_existing:
        path_fn.write_text(_FIRE_CONFIG_TEMPLATE_PATH_FN.expanduser().read_text())
        logger.info(f'Copied template fire config file to "{path_fn}" from '
                    f'"{_FIRE_CONFIG_TEMPLATE_PATH_FN}" (replace_existing={replace_existing})')

# Return same fire config data from previous call - dones't work with dict args
# @lru_cache(maxsize=2, typed=False)
def get_user_fire_config(path=None, fn=None, base_paths=(), use_global=True):
    """

    Args:
        path         : Path to config file
        fn           : Config filename
        base_paths   : Base paths to substitute into path

    Returns: (fire_config, config_groups, path_fn)

    """
    logger.debug(f'path={path}, fn={fn}, base_paths={base_paths}, use_global={use_global}')
    if (fire.config is None) or (not use_global):

        base_paths = dict(base_paths)
        config_groups = {}

        fire_user_path = get_fire_user_directory(fire_user_dir=base_paths.get(_FIRE_USER_DIR_KEY, path))

        path_fn = get_fire_user_config_path_fn(path=base_paths.get(_FIRE_USER_DIR_KEY), fn=fn)

        base_paths.setdefault(_FIRE_USER_DIR_KEY, fire_user_path)

        # Copy default fire_config.json file from FIRE repo if not found
        if not path_fn.is_file():
            copy_default_user_settings()

        # Read raw contents of file
        fire_config = _read_user_fire_config()

        # Substitute base paths into user paths for simpler subsequent use (makes easy to change fire user path at runtime)
        base_paths, fire_config = update_base_paths_in_fire_config(fire_config, base_paths_user=base_paths,
                                                                   update_inplace=True)

        config_groups['fire_paths'] = base_paths

        fire.config = fire_config
        fire.config_groups = config_groups
        fire.base_paths = base_paths
        fire.config_path_fn = path_fn
    else:
        fire_config = fire.config
        config_groups = fire.config_groups
        path_fn = fire.config_path_fn

    UserConfig = namedtuple('UserConfig', ('fire_config', 'config_groups', 'path_fn'))

    return UserConfig(fire_config=fire_config, config_groups=config_groups, path_fn=path_fn)


if __name__ == '__main__':
    copy_default_user_settings(replace_existing=True)
    pass
