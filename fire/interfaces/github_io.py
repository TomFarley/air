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
logger.propagate = False

import os
import subprocess
from python.settings import FPATH_TOP_LOG, FPATH_TOP_LOG_REMOTE, YES, NO


def update_remote_log(logger=None):
    """ push top-level MWI log file to github """

    try:
        out = subprocess.run(['tail', '-n', '200', FPATH_TOP_LOG], stdout=subprocess.PIPE, ).stdout.decode()
        out = out.replace('\n', '<br>')
        with open(FPATH_TOP_LOG_REMOTE, 'w') as f:
            f.write(out)

        out = subprocess.run(['git', '-C', os.path.dirname(FPATH_TOP_LOG_REMOTE), 'commit', '-am', 'auto-update', ],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,)
        # r = 'https://MultiWavelengthImaging:gomwi2019@github.com/MultiWavelengthImaging/MultiWavelengthImaging.github.io'
        r = 'https://MultiWavelengthImaging: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@github.com/MultiWavelengthImaging/MultiWavelengthImaging.github.io'
        out = subprocess.run(['git', '-C', os.path.dirname(FPATH_TOP_LOG_REMOTE), 'push', '-q', r, 'master'],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,)
        if logger is not None:
            logger.info('update_remote_log: ' + YES)

        return 0

    except Exception as e:

        if logger is not None:
            logger.info('update_remote_log: ' + NO)
            logger.info('update_remote_log: ' + repr(e))
            return 1
        else:
            raise

def update_top_log_remote():

    # update file
    out = subprocess.run(['tail', '-n', '200', FPATH_TOP_LOG], stdout=subprocess.PIPE, ).stdout.decode()
    out = out.replace('\n', '<br>')
    with open(FPATH_TOP_LOG_REMOTE, 'w') as f:
        f.write(out)

    # push to github
    out = subprocess.run(['git', '-C', os.path.dirname(FPATH_TOP_LOG_REMOTE), 'commit', '-am', 'auto-update', ],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,)
    # r = 'https://MultiWavelengthImaging:gomwi2019@github.com/MultiWavelengthImaging/MultiWavelengthImaging.github.io'
    r = 'https://MultiWavelengthImaging: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@github.com/MultiWavelengthImaging/MultiWavelengthImaging.github.io'
    out = subprocess.run(['git', '-C', os.path.dirname(FPATH_TOP_LOG_REMOTE), 'push', '-q', r, 'master'],
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,)


if __name__ == '__main__':
    pass