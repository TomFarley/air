#!/usr/bin/env python

"""


Created: 
"""

import logging, datetime
from typing import Union, Iterable, Sequence, Tuple, Optional, Any, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from fire.interfaces.uda_utils import get_uda_client

from pyEquilibrium.equilibrium import equilibrium
from mastu_exhaust_analysis import mastu_equilibrium

logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.DEBUG)

def read_bfield_data(shot, r, z, t, epm_path='/common/uda-scratch/lkogan/efitpp_eshed/epm0{shot}.nc'):
    uda_module, client = get_uda_client()

    epm_fn = epm_path.format(shot=shot, pulse=shot)
    epm_times = client.get('/epm/time', epm_fn).data
    converged_status = client.get('/epm/equilibriumStatusInteger', epm_fn)
    mask_converged = converged_status.data == 1
    t_converged = epm_times[mask_converged]

    print(f't requested: {t}')
    print(f't converged: {t_converged}')

    logger.info(f'Reading equilibrium data for shot {shot}')

    b = xr.DataArray(data=np.full((len(t), len(r), 3), np.nan), dims=('t', 'path', 'B_vec'),
                     coords={'t': t, 'path': np.arange(len(r)), 'B_vec': ['r', 'z', 'phi']}, name='B_field')

    t_start = datetime.datetime.now()
    for ti in t:
        if np.any(np.isclose(ti,t_converged)):
            eqm_data = equilibrium(shot=epm_fn, device='MASTU', time=ti)

            for i, (ri, zi) in enumerate(zip(r, z)):
                # Magnetic field data at this point in space
                br = eqm_data.BR(ri, zi)[0][0]
                bz = eqm_data.BZ(ri, zi)[0][0]
                bt = eqm_data.Bt(ri, zi)[0][0]

                b.loc[dict(t=ti, path=i, B_vec=['r', 'z', 'phi'])] = np.array([br, bz, bt])
            logger.info(f'Read equilibrium data for t={ti}')
            print(f'Read equilibrium data for t={ti}')
        else:
            logger.warning(f'{ti} not in efit converged times')
    t_end = datetime.datetime.now()

    # TODO: Interpolate nans

    logger.info(f'Read equilibrium data for shot {shot} in {t_end-t_start}')
    print(f'Read equilibrium data for shot {shot} in {t_end-t_start}')

    return b



def lookup_b_field(gfile, ):
    equil = mastu_equilibrium()

    equil.load_efitpp('epm044427.nc', time=0.5)


if __name__ == '__main__':
    shot = 44345
    r = np.linspace(1, 1.3, 31)
    z = np.linspace(-1.3, -1.5, 21)
    t = np.linspace(0.05, 0.3, 26)
    b = read_bfield_data(shot, r, z, t)

    print(b)
    pass