#!/usr/bin/env python

"""


Created: 
"""

import logging
from typing import Union, Iterable, Sequence, Tuple, Optional, Any, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from fire.misc.utils import filter_kwargs

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# ===================== PLUGIN MODULE ATTRIBUTES =====================
# Required:
output_format_plugin_name = 'uda_putdata'
# Optional:
output_filename_format = '{diag_tag}{shot:06d}.nc'  # Filename of output
output_path_format = '{fire_path}/../'  # Path to save output
# See bottom of file for function aliases
# ====================================================================

not_set = object()

# Updated IDL sched code output in /home/athorn/IR/Latest/sched_air_netcdf/src/netcdfout.pro
IDL_NETCDF_LABELS = ['Time', 'Value', 'Min q', 'Max q', 'Min q (ELM alpha)', 'Max q (ELM alpha)', 'R(peak q)', 'P(tot)',
    'q_profile', 't_profile', 'q_profile_elm', 'Num sat. pix.', 'R coord', 'P(tot)', 'E(tot)', 'E_sum(tot)',
    'P(tot) ELM', 'E(tot) ELM', 'E_sum(tot) ELM', 'Max temp.', 'Alpha', 'Alpha ELM', 'r_start', 'phi_start',
    'z_start', 'r_extent', 'phi_extent', 'z_extent']

output_signals = {
    "AIR_ALPHACONST_ISP": {"label": 'alpha', "units": "N/A", "dtype": float,
                           "description": "Surface layer alpha constant for inner strike point used in THEODOR"},
    "AIR_ALPHACONST_ISP_ELM": {"label": 'alpha', "units": "N/A", "dtype": float,
                "description": "Surface layer alpha constant for inner strike point used in THEODOR during ELMS"},
    "AIR_ALPHACONST_OSP": {"label": 'alpha', "units": "N/A", "dtype": float,
                     "description": "Surface layer alpha constant for outer strike point used in THEODOR"},
    "AIR_ALPHACONST_OSP_ELM": {"label": 'alpha', "units": "N/A", "dtype": float,
                     "description": "Surface layer alpha constant for outer strike point used in THEODOR during ELMS"},
    "AIR_CAMERA VIEW_ISP": {"label": "View index", "units": "N/A", "dtype": int,
                            "description": "Integer index of camera view for inner strike point"},
    "AIR_CAMERA VIEW_OSP": {"label": "View index", "units": "N/A", "dtype": int,
                            "description": "Integer index of camera view for outer strike point"},
    "AIR_ERRORFLAG": {"label": "Error flag", "units": "N/A", "dtype": int,
                            "description": "Error state from analysis"},
    "AIR_ETOTSUM_ISP": {"label": not_set, "units": 'kJ', "dtype": np.ndarray, "description": not_set},
    "AIR_ETOTSUM_ISP_ELM": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_ETOTSUM_OSP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_ETOTSUM_OSP_ELM": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_ETOT_ISP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_ETOT_ISP_ELM": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_ETOT_OSP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_ETOT_OSP_ELM": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_LAMPOWPP_ISP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_LAMPOWPP_OSP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_LAMPOWSOL_ISP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_LAMPOWSOL_OSP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_MINPOWER_DENSITY_ISP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_MINPOWER_DENSITY_OSP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_MINPOW_DENS_ISP_ELM": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_MINPOW_DENS_OSP_ELM": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_PASSNUMBER": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_PEAKPOWER_POS_OSP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_PHI_EXTENT_ISP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_PHI_EXTENT_OSP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_PHI_START_ISP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_PHI_START_OSP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_PKPOWER_DENSITY_ISP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_PKPOWER_DENSITY_OSP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_PKPOWER_POS_ISP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_PKPOW_DENS_ISP_ELM": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_PKPOW_DENS_OSP_ELM": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_PTOT_ISP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_PTOT_ISP_ELM": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_PTOT_OSP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_PTOT_OSP_ELM": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_QPROFILE_ISP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_QPROFILE_ISP_ELM": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_QPROFILE_OSP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_QPROFILE_OSP_ELM": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_RCOORD_ISP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_RCOORD_OSP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_R_EXTENT_ISP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_R_EXTENT_OSP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_R_START_ISP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_R_START_OSP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_SATPIXELS_ISP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_SATPIXELS_OSP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_STATUS": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_TEMPERATURE_ISP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_TEMPERATURE_OSP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_TPROFILE_ISP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_TPROFILE_OSP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_Z_EXTENT_ISP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_Z_EXTENT_OSP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_Z_START_ISP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set},
    "AIR_Z_START_OSP": {"label": not_set, "units": not_set, "dtype": not_set, "description": not_set}
}


def write_processed_ir_to_uda_netcdf_file(path_data, image_data, path_names,
                                          variable_names_path, variable_names_time, variable_names_image,
                                          header_info, device_info, meta_data,
                                          fn_output='{diag_tag}{shot:06d}.nc', path_output='./'):
    """
    NETCDF output code from /home/athorn/IR/Latest/sched_air_netcdf/src/netcdfout.pro:
    Returns:

    - Using puddata in development:
    module purge
    module load FUN
module unload uda
module use /projects/UDA/uda-install-develop/modulefiles
module load uda/develop              #-fatclient

    """
    from fire.interfaces.uda_utils import (putdata_create, putdata_device, putdata_variables_from_datasets,
                                           putdata_close)
    # Set up output file
    file_id, path_fn = putdata_create(fn=fn_output, path=path_output, close=False, **{**header_info, **meta_data})

    # Write device information
    device_name = meta_data['diag_tag']
    putdata_device(device_name, device_info=device_info)
    pass


    variable_meta_data = meta_data.get('variables', {})
    # Write dimensions, coordinates, data and additional attributes from xarray.Dataset objects
    putdata_variables_from_datasets(path_data, image_data, path_names,
                                    variable_names_path, variable_names_time, variable_names_image)

    # TODO: Include mapping from old MAST signal names to new signal paths?

    # Close file
    putdata_close(file_id=None)

    if not path_fn.is_file():
        # raise FileNotFoundError(f'UDA output file does not exist: {path_fn}')
        logger.exception(f'UDA output file does not exist: {path_fn}')
        success = False
    else:
        success = True

    return dict(success=success, path_fn=path_fn)

# ================== PLUGIN MODULE FUNCTION ALIASES ==================
write_output_file = write_processed_ir_to_uda_netcdf_file
# ====================================================================

if __name__ == '__main__':
    pass