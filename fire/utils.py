#!/usr/bin/env python

"""Miscelanious utility functions

Created: 11-10-19
"""

import logging
from typing import Union, Iterable, Tuple, Optional
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def update_call_args(user_defaults, pulse, camera, machine):
    """Replace 'None' values with user's preassigned default values

    :param user_defaults: Dict of user's default settings
    :param pulse: Shot/pulse number or string name for synthetic movie data
    :param camera: Name of camera to analyse (unique name of camera or diagnostic code)
    :param machine: Tokamak that the data originates from
    :return:
    """
    if pulse is None:
        pulse = user_defaults['pulse']
    if camera is None:
        camera = user_defaults['camera']
    if machine is None:
        machine = user_defaults['machine']
    return pulse, camera, machine

if __name__ == '__main__':
    pass