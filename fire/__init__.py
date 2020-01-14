import logging
from pathlib import Path

import pandas as pd
import xarray as xr

# Set up logger for module
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Store important paths for quick reference
fire_paths = {'root': Path(__file__).parent}
fire_paths['input_files'] = fire_paths['root'] / 'input_files'
fire_paths['config'] = Path('~/.fire_config.json').expanduser()

if not fire_paths['config'].exists():
    # Set up fire config file
    # TODO: skip if on scheduler?
    fire_config_template_path_fn = fire_paths['input_files'] / 'user' / 'fire_config.json'
    fire_paths['config'].write_text(fire_config_template_path_fn.read_text())
    print(f'Coppied template fire config to "{fire_paths["config"]}" from "{fire_config_template_path_fn}"')

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
# from .theodor import theo_mul_33 as theodor

# from .. import tests
# from ..tests impoort unit