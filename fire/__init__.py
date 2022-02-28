# -*- coding: future_fstrings -*-
import logging as _logging
import logging.config
import yaml as _yaml
from pathlib import Path as _Path
import os as _os

import pandas as _pd
import xarray as _xr
import matplotlib.pyplot as _plt

# Single central version number that is pulled in by setup.py etc
__version__ = "2.0.0"

PATH_FIRE_SOURCE = _Path(__file__).parent
PATH_AIR_REPO = PATH_FIRE_SOURCE.parent

# Instantiate a few global variables that will be set in fire func calls (only use these 4!)
user_path = None  # Path to users fire directory, set in call to fire.interfaces.user_config.get_fire_user_directory()
config = None  # Set by call to fire.interfaces.user_config.get_user_fire_config()
config_groups = None  # Set by call to fire.interfaces.user_config.get_user_fire_config()
config_path_fn = None  # Set by call to fire.interfaces.user_config.get_user_fire_config()
base_paths = None  # Set by call to fire.interfaces.user_config.get_user_fire_config()
active_machine_plugin = None  # Set by call to fire.plugins.plugins_machine.get_machine_plugins()
active_calcam_calib = None

def substritute_tilde_for_home(path):
    path = _Path(str(path).replace(str(_Path.home()), '~'))  # Contract home
    return path

def setup_fire_logger():
    # Set up logger for module
    _logging.basicConfig()
    logger_fire = _logging.getLogger(__name__)

    fn = (PATH_FIRE_SOURCE / 'logging_config.yaml').expanduser()

    with open(fn, 'r') as f:
        _logging_config = _yaml.safe_load(f.read())
        # Make sure relative paths are relative to fire root directory
        for handler in _logging_config['handlers']:
            if 'filename' in _logging_config['handlers'][handler]:
                fn_log = _Path(_logging_config['handlers'][handler]['filename'].format(
                    fire=PATH_FIRE_SOURCE.expanduser())).expanduser().resolve()
                fn_log.parent.mkdir(exist_ok=True, parents=True)  # create tmp/log dir which is otherwise in .gitignore
                _logging_config['handlers'][handler]['filename'] = fn_log
        _logging.config.dictConfig(_logging_config)

    # Set logging level for console output handler propagated throughout fire package
    handlers = logger_fire.handlers
    for handler in handlers:
        logger_fire.debug(f'FIRE logging handler {handler} set to level: {handler.level}')

    if len(handlers) > 0:
        stream_handler = handlers[0]
    else:
        logger_fire.warning(f'Failed to set up fire stream handler')

    logger_fire.setLevel(_logging.DEBUG)
    print(f'FIRE parent logger {logger_fire} set to level: {logger_fire.level}')

    # [i.setLevel('DEBUG') for i in [logger_fire, stream_handler]]
    # stream_handler.setLevel(_logging.DEBUG)  # Uncomment to print debug level messages to console throughout fire

def configure_libarary_defaults():
    # Update pandas display settings
    # Double max width of displayed output in terminal so doesn't wrap over many lines
    _pd.set_option("display.width", 160)  # TODO: Use None when in ipython terminal - auto size?
    # Double max column display width to display long descriptive strings in columns
    _pd.set_option("display.max_colwidth", 80)
    _pd.options.display.max_rows = 60
    _pd.options.display.max_columns = 500

    # Update xarray display settings
    _xr.set_options(**{'cmap_sequential': 'gray'})
    _xr.set_options(display_max_rows=100)
    _xr.set_options(display_width=100)  # Default 80
    _xr.set_options(keep_attrs=True)
    _xr.set_options(display_expand_coords='default')

    # Use custom mpl style
    _plt.style.use(f'{PATH_FIRE_SOURCE}/plotting/fire.mplstyle')

setup_fire_logger()

configure_libarary_defaults()

PATH_FIRE_SOURCE = substritute_tilde_for_home(PATH_FIRE_SOURCE)
PATH_AIR_REPO = substritute_tilde_for_home(PATH_AIR_REPO)

# copy_default_user_settings(replace_existing=False)

# Read in user config file here for access throughout
# user_config = _read_user_fire_config(fire_paths['config'])

# Set up UDA client once here for use through fire - single client object
# uda_module, client = get_uda_client(use_mast_client=False, try_alternative=True)


# Import FIRE sub-modules
# from .interfaces import interfaces
from .misc import utils
from .interfaces import user_config
from .interfaces.user_config import get_user_fire_config, get_fire_user_directory