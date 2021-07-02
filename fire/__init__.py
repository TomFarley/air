# -*- coding: future_fstrings -*-
import logging as _logging
import logging.config
import yaml as _yaml
from pathlib import Path as _Path
import os as _os

import pandas as _pd
import xarray as _xr
import matplotlib.pyplot as _plt

# Import functions useful at top level
from .interfaces.read_user_fire_config import read_user_fire_config

# from .theodor import theo_mul_33 as theodor

# from .. import tests
# from ..tests import unit

# Single central version number that is pulled in by setup.py etc
__version__ = "2.0.0"

# Set up logger for module
_logging.basicConfig()
logger_fire = _logging.getLogger(__name__)

_fire_user_dir_env_var = 'FIRE_USER_DIR'
_fire_user_dir = _os.environ.get(_fire_user_dir_env_var, None)
if _fire_user_dir is None:
    _fire_user_dir = "~/.fire/"
    print(f'- Environment variable "{_fire_user_dir_env_var}" has not been set. Using default path '
          f'"{_fire_user_dir}" for configuration file and user outputs.\n'
          f'- To set your "{_fire_user_dir_env_var}" environment variable (on linux) do:\n'
          f'- $ export {_fire_user_dir_env_var}="<my_path>"  # It is recommended to add this to your .bashrc')
    _os.environ[_fire_user_dir_env_var] = str(_fire_user_dir)  # visible in this process + all children
else:
    _logging.info(f'Using FIRE user directory "{_fire_user_dir}" from env var {_fire_user_dir_env_var}')
_fire_user_dir = _Path(_fire_user_dir).expanduser()

# Store important paths for quick reference
# TODO: Read template json config file and in turn look up user config file if possible and use to set user fire path
fire_paths = {'root': _Path(__file__).parent}
fire_paths['user_files'] = _fire_user_dir
fire_paths['input_files'] = fire_paths['root'] / 'input_files'
fire_paths['config'] = _fire_user_dir / '.fire_config.json'

with open(fire_paths['root'] / 'logging_config.yaml', 'r') as f:
    _logging_config = _yaml.safe_load(f.read())
    # Make sure relative paths are relative to fire root directory
    for handler in _logging_config['handlers']:
        if 'filename' in _logging_config['handlers'][handler]:
            _logging_config['handlers'][handler]['filename'] = _logging_config['handlers'][handler]['filename'].format(fire=fire_paths[
                'root'])
    _logging.config.dictConfig(_logging_config)

# Set logging level for console output handler propagated throughout fire package
handlers = logger_fire.handlers
if len(handlers) > 0:
    stream_handler = handlers[0]
else:
    logger_fire.warning(f'Failed to set up fire stream handler')
logger_fire.setLevel(_logging.DEBUG)
# stream_handler.setLevel(_logging.DEBUG)  # Uncomment to print debug level messages to console throughout fire

active_machine_plugin = None
active_calcam_calib = None

def copy_default_user_settings(replace_existing=False):
    # Set up fire config file
    # TODO: skip if on scheduler?
    _fire_config_template_path_fn = fire_paths['input_files'] / 'user' / 'fire_config.json'
    if (not fire_paths['config'].exists()) or replace_existing:
        fire_paths['config'].write_text(_fire_config_template_path_fn.read_text())
        logger_fire.info(f'Copied template fire config file to "{fire_paths["config"]}" from '
                    f'"{_fire_config_template_path_fn}"')

copy_default_user_settings(replace_existing=False)

# Read in user config file here for access throughout
user_config = read_user_fire_config(fire_paths['config'])

fire_paths['user'] = _Path(user_config['user']['directory']).expanduser()

# Update pandas display settings
# Double max width of displayed output in terminal so doesn't wrap over many lines
_pd.set_option("display.width", 160)  # TODO: Use None when in ipython terminal - auto size?
# Double max column display width to display long descriptive strings in columns
_pd.set_option("display.max_colwidth", 80)
_pd.options.display.max_rows = 60
_pd.options.display.max_columns = 500

_xr.set_options(**{'cmap_sequential': 'gray'})

# Use custom mpl style
_plt.style.use(f'{fire_paths["root"]}/plotting/fire.mplstyle')

# Import FIRE sub-modules
from .interfaces import interfaces
from .misc import utils