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

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

class FireException(Exception):
    """Base exception for FIRE related errors"""
    msg_format = 'Error in IR analysis'
    def __init__(self, msg=None, original_exception=None, info=None):
        if msg is not None:
            self.msg_format = msg
        if info is None:
            info = {}
        self.original_exception = original_exception
        self.info = info
        msg = self.format_msg()
        super().__init__(msg)

    def format_msg(self):
        msg = self.msg_format.format(**self.info)
        if self.original_exception is not None:
            msg += f': {self.original_exception}'
        return msg

class InputFileException(FireException, ValueError):
    """Error related to reading FIRE input files"""
    msg_format = 'Failed to read FIRE input file "{fn}"'

if __name__ == '__main__':
    pass