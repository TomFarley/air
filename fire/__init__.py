# -*- coding: future_fstrings -*-
import logging
import logging.config
import yaml
from pathlib import Path

import pandas as pd
import xarray as xr
# Set up logger for module
logging.basicConfig(
    # format="%(levelname)s:%(module)s:%(funcName)s:%(lineno)s: %(message)s",
    )

with open('logging_config.yaml', 'r') as f:
    config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

logger_fire = logging.getLogger(__name__)
logger_fire.setLevel(logging.DEBUG)

# Store important paths for quick reference
fire_paths = {'root': Path(__file__).parent}
fire_paths['input_files'] = fire_paths['root'] / 'input_files'
fire_paths['config'] = Path('~/.fire_config.json').expanduser()

active_machine_plugin = None
active_calcam_calib = None

def copy_default_user_settings(replace_existing=False):
    # Set up fire config file
    # TODO: skip if on scheduler?
    fire_config_template_path_fn = fire_paths['input_files'] / 'user' / 'fire_config.json'
    if (not fire_paths['config'].exists()) or replace_existing:
        fire_paths['config'].write_text(fire_config_template_path_fn.read_text())
        logger_fire.info(f'Copied template fire config file to "{fire_paths["config"]}" from '
                    f'"{fire_config_template_path_fn}"')

copy_default_user_settings(replace_existing=False)

# Update pandas display settings
# Double max width of displayed output in terminal so doesn't wrap over many lines
pd.set_option("display.width", 160)  # TODO: Use None when in ipython terminal - auto size?
# Double max column display width to display long descriptive strings in columns
pd.set_option("display.max_colwidth", 80)
pd.options.display.max_rows = 60
pd.options.display.max_columns = 500

xr.set_options(**{'cmap_sequential': 'gray'})

# Import FIRE sub-modules
from .interfaces import interfaces
from.misc import utils
# from .theodor import theo_mul_33 as theodor

# from .. import tests
# from ..tests impoort unit