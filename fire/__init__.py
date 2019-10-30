from pathlib import Path

import pandas as pd

# Store important paths for quick reference
fire_paths = {'root': Path(__file__).parent}
fire_paths['input_files'] = fire_paths['root'] / 'input_files'
fire_paths['user_inputs'] = fire_paths['input_files'] / 'user'


# Update pandas display settings
# Double max width of displayed output in terminal so doesn't wrap over many lines
pd.set_option("display.width", 160)  # TODO: Use None when in ipython terminal - auto size?
# Double max column display width to display long descriptive strings in columns
pd.set_option("display.max_colwidth", 80)
pd.options.display.max_rows = 60
pd.options.display.max_columns = 500
