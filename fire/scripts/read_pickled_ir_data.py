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
from fire.misc.utils import make_iterable
from fire.plugins.output_format_plugins.pickle_output import read_output_file

logger = logging.getLogger(__name__)

def read_data_for_pulses_pickle(diag_tag_raw: str, pulses: dict, machine:str= 'mast_u', generate=True, recompute=False):
    data = {}

    pulse_values = list(pulses.keys()) if isinstance(pulses, dict) else make_iterable(pulses)

    for pulse in pulse_values:
        if isinstance(pulse, (tuple, list)):  # When pulse is passed in tuple with dict of kwargs for that pulse
            pulse = pulse[0]
        pulse = int(pulse)
        if not recompute:
            try:
                data[pulse], fn_pickle = read_output_file(diag_tag_raw, pulse, machine=machine)
            except FileNotFoundError as e:
                exception = e
                success = False
                recompute = True
                logger.warning(e)
            except AttributeError as e:  # Pickled library out of date
                exception = e
                success = False
                recompute = True
                logger.warning('Pickled object library likely updated? Need to recompute?')
                logger.warning(e)
            else:
                success = True
                logger.info(f'Restored picked analysed output from: {fn_pickle}')
        if (recompute or ((not success) and generate)):
            # debug = {'movie_intensity_stats': True}
            from fire.scripts import scheduler_workflow
            debug = {}
            outputs = scheduler_workflow.scheduler_workflow(pulse=pulse, camera=diag_tag_raw, pass_no=0, machine=machine,
                                                            scheduler=True, debug=debug)
            scheduler_workflow.copy_uda_netcdf_output(outputs, copy_to_uda_scrach=True, clean_netcdf=True)

            data[pulse], fn_pickle = read_output_file(diag_tag_raw, pulse, machine=machine)
        elif (not success) and (not generate):
            raise exception

    logger.info(f'Read data for shots: {pulse_values}')
    return data

if __name__ == '__main__':
    pass