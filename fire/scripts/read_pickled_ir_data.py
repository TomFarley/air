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
logger.propagate = False

if __name__ == '__main__':
    pass


def read_data_for_pulses_pickle(camera: str, pulses: dict, machine:str= 'mast_u', generate=True, recompute=False):
    data = {}

    pulse_values = list(pulses.keys()) if isinstance(pulses, dict) else make_iterable(pulses)

    for pulse in pulse_values:
        if isinstance(pulse, (tuple, list)):  # When pulse is passed in tuple with dict of kwargs for that pulse
            pulse = pulse[0]
        pulse = int(pulse)
        if not recompute:
            try:
                data[pulse] = read_output_file(camera, pulse, machine=machine)
                success = True
            except FileNotFoundError as e:
                exception = e
                success = False
                recompute = True
                logger.warning(e)
        if (recompute or ((not success) and generate)):
            # debug = {'movie_intensity_stats': True}
            from fire.scripts import scheduler_workflow
            debug = {}
            out = scheduler_workflow.scheduler_workflow(pulse=pulse, camera=camera, pass_no=0, machine=machine,
                                                  scheduler=False, debug=debug)
            data[pulse] = read_output_file(camera, pulse, machine=machine)
        elif (not success) and (not generate):
            raise exception

    logger.info(f'Read data for shots: {pulse_values}')
    return data