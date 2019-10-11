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

def update_call_args(user_defaults, shot, camera, machine):
    """Replace 'None' values with user's preassigned default values

    :param user_defaults: Dict of user's default settings
    :param shot: Shot/pulse number or string name for synthetic movie data
    :param camera: Name of camera to analyse (unique name of camera or diagnostic code)
    :param machine: Tokamak that the data originates from
    :return:
    """
    if shot is None:
        shot = user_defaults['shot']
    if camera is None:
        camera = user_defaults['camera']
    if machine is None:
        machine = user_defaults['machine']
    return shot, camera, machine

if __name__ == '__main__':
    pass