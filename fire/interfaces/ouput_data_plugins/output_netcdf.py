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
logger.setLevel(logging.DEBUG)

def write_processed_ir_to_netcdf():
    """"""


    """
    alphaconst_isp
    alphaconst_isp_elm
    alphaconst_osp
    alphaconst_osp_elm
    
    client.list(pyuda.ListType.SIGNALS, alias='air')
    
'AIR_ALPHACONST', generic_n
, mds_name='\\TOP.ANALYSED.
'AIR_ALPHACONST_ISP', gener
'AIR_ALPHACONST_ISP_ELM', g
'AIR_ALPHACONST_OSP', gener
'AIR_ALPHACONST_OSP_ELM', g
'AIR_CAMERA VIEW', generic_
'AIR_CAMERA VIEW_ISP', gene
'AIR_CAMERA VIEW_OSP', gene
'AIR_ENERGY', generic_name=
'AIR_ENERGY_30K', generic_n
'AIR_ERRORFLAG', generic_na
'AIR_ETOT', generic_name=''
'AIR_ETOTSUM', generic_name
'AIR_ETOTSUM_ISP', generic_
'AIR_ETOTSUM_ISP_ELM', gene
'AIR_ETOTSUM_OSP', generic_
'AIR_ETOTSUM_OSP_ELM', gene
'AIR_ETOT_ISP', generic_nam
'AIR_ETOT_ISP_ELM', generic
'AIR_ETOT_OSP', generic_nam
'AIR_ETOT_OSP_ELM', generic
'AIR_LAMPOWPP', generic_nam
'AIR_LAMPOWPP2', generic_na
'AIR_LAMPOWPP_ISP', generic
'AIR_LAMPOWPP_OSP', generic
'AIR_LAMPOWSOL', generic_na
'AIR_LAMPOWSOL2', generic_n
'AIR_LAMPOWSOL_ISP', generi
'AIR_LAMPOWSOL_OSP', generi
'AIR_MINPOWER_DENSITY', gen
'AIR_MINPOWER_DENSITY_30K',
'AIR_MINPOWER_DENSITY_ISP',
'AIR_MINPOWER_DENSITY_OSP',
'AIR_MINPOW_DENS_ISP_ELM', 
'AIR_MINPOW_DENS_OSP_ELM', 
'AIR_PASSNUMBER', generic_n
'AIR_PEAKPOWER_DENSITY', ge
'AIR_PEAKPOWER_DENSITY_30',
'AIR_PEAKPOWER_POS', generi
'AIR_PEAKPOWER_POS_ISP', ge
'AIR_PEAKPOWER_POS_OSP', ge
'AIR_PHI_EXTENT', generic_n
'AIR_PHI_EXTENT_ISP', gener
'AIR_PHI_EXTENT_OSP', gener
    
    
    
    
    [
 ListData(shot=0, pass_=-1, signal_name='AIR_ALPHACONST', generic_name='', source_alias='air', type='Analysed', 
 description='Replaced', signal_status=1, mds_name='\\TOP.ANALYSED.AIR:ALPHACONST'),
 ListData(shot=0, pass_=-1, signal_name='AIR_ALPHACONST_ISP', generic_name='', source_alias='air', type='Analysed', description='Surface layer coefficient applied to inner strike point', signal_status=1, mds_name='\\TOP.ANALYSED.AIR.ALPHACONST_:ISP'),
 ListData(shot=0, pass_=-1, signal_name='AIR_ALPHACONST_ISP_ELM', generic_name='', source_alias='air', type='Analysed', description='Surface layer coefficient applied to inner strike point for ELMs', signal_status=1, mds_name='\\TOP.ANALYSED.AIR.ALPHACONST_:ISP_ELM'),
 ListData(shot=0, pass_=-1, signal_name='AIR_ALPHACONST_OSP', generic_name='', source_alias='air', type='Analysed', description='Surface layer coefficient applied to outer strike point', signal_status=1, mds_name='\\TOP.ANALYSED.AIR.ALPHACONST_:OSP'),
 ListData(shot=0, pass_=-1, signal_name='AIR_ALPHACONST_OSP_ELM', generic_name='', source_alias='air', type='Analysed', description='Surface layer coefficient applied to outer strike point for ELMs', signal_status=1, mds_name='\\TOP.ANALYSED.AIR.ALPHACONST_:OSP_ELM'),
 ListData(shot=0, pass_=-1, signal_name='AIR_CAMERA VIEW', generic_name='', source_alias='air', type='Analysed', description='Replaced', signal_status=1, mds_name='\\TOP.ANALYSED.AIR:CAMERA_VIEW'),
 ListData(shot=0, pass_=-1, signal_name='AIR_CAMERA VIEW_ISP', generic_name='', source_alias='air', type='Analysed', description='', signal_status=1, mds_name='\\TOP.ANALYSED.AIR.CAMERA_VIEW_:ISP'),
 ListData(shot=0, pass_=-1, signal_name='AIR_CAMERA VIEW_OSP', generic_name='', source_alias='air', type='Analysed', description='', signal_status=1, mds_name='\\TOP.ANALYSED.AIR.CAMERA_VIEW_:OSP'),
 ListData(shot=0, pass_=-1, signal_name='AIR_ENERGY', generic_name='', source_alias='air', type='Analysed', description='No longer used', signal_status=1, mds_name='\\TOP.ANALYSED.AIR:ENERGY'),
 ListData(shot=0, pass_=-1, signal_name='AIR_ENERGY_30K', generic_name='', source_alias='air', type='Analysed', description='No longer used', signal_status=1, mds_name='\\TOP.ANALYSED.AIR:ENERGY_30K'),
 ListData(shot=0, pass_=-1, signal_name='AIR_ERRORFLAG', generic_name='', source_alias='air', type='Analysed', description='', signal_status=1, mds_name='\\TOP.ANALYSED.AIR:ERRORFLAG'),
 ListData(shot=0, pass_=-1, signal_name='AIR_ETOT', generic_name='', source_alias='air', type='Analysed', description='Replaced', signal_status=1, mds_name='\\TOP.ANALYSED.AIR:ETOT'),
 ListData(shot=0, pass_=-1, signal_name='AIR_ETOTSUM', generic_name='', source_alias='air', type='Analysed', description='Replaced', signal_status=1, mds_name='\\TOP.ANALYSED.AIR:ETOTSUM'),
 ListData(shot=0, pass_=-1, signal_name='AIR_ETOTSUM_ISP', generic_name='', source_alias='air', type='Analysed', description='Cumulative energy to divertor inner', signal_status=1, mds_name='\\TOP.ANALYSED.AIR:ETOTSUM_ISP'),
 ListData(shot=0, pass_=-1, signal_name='AIR_ETOTSUM_ISP_ELM', generic_name='', source_alias='air', type='Analysed', description='Cumulative energy to divertor inner for ELM', signal_status=1, mds_name='\\TOP.ANALYSED.AIR.ETOTSUM_ISP_:ELM'),
 ListData(shot=0, pass_=-1, signal_name='AIR_ETOTSUM_OSP', generic_name='', source_alias='air', type='Analysed', description='Cumulative energy to divertor outer', signal_status=1, mds_name='\\TOP.ANALYSED.AIR:ETOTSUM_OSP'),
 ListData(shot=0, pass_=-1, signal_name='AIR_ETOTSUM_OSP_ELM', generic_name='', source_alias='air', type='Analysed', description='Cumulative energy to divertor outer for ELM', signal_status=1, mds_name='\\TOP.ANALYSED.AIR.ETOTSUM_OSP_:ELM'),
 ListData(shot=0, pass_=-1, signal_name='AIR_ETOT_ISP', generic_name='', source_alias='air', type='Analysed', description='Instantaneous energy to divertor inner', signal_status=1, mds_name='\\TOP.ANALYSED.AIR:ETOT_ISP'),
 ListData(shot=0, pass_=-1, signal_name='AIR_ETOT_ISP_ELM', generic_name='', source_alias='air', type='Analysed', description='Instantaneous energy to divertor inner for ELM', signal_status=1, mds_name='\\TOP.ANALYSED.AIR:ETOT_ISP_ELM'),
 ListData(shot=0, pass_=-1, signal_name='AIR_ETOT_OSP', generic_name='', source_alias='air', type='Analysed', description='Instantaneous energy to divertor outer', signal_status=1, mds_name='\\TOP.ANALYSED.AIR:ETOT_OSP'),
 ListData(shot=0, pass_=-1, signal_name='AIR_ETOT_OSP_ELM', generic_name='', source_alias='air', type='Analysed', description='Instantaneous energy to divertor outer for ELM', signal_status=1, mds_name='\\TOP.ANALYSED.AIR:ETOT_OSP_ELM'),
 ListData(shot=0, pass_=-1, signal_name='AIR_LAMPOWPP', generic_name='', source_alias='air', type='Analysed', description='No longer written out', signal_status=1, mds_name='\\TOP.ANALYSED.AIR:LAMPOWPP'),
 ListData(shot=0, pass_=-1, signal_name='AIR_LAMPOWPP2', generic_name='', source_alias='air', type='Analysed', description='No longer written out', signal_status=1, mds_name='\\TOP.ANALYSED.AIR:LAMPOWPP2'),
 ListData(shot=0, pass_=-1, signal_name='AIR_LAMPOWPP_ISP', generic_name='', source_alias='air', type='Analysed', description='No longer written out', signal_status=1, mds_name='\\TOP.ANALYSED.AIR:LAMPOWPP_ISP'),
 ListData(shot=0, pass_=-1, signal_name='AIR_LAMPOWPP_OSP', generic_name='', source_alias='air', type='Analysed', description='No longer written out', signal_status=1, mds_name='\\TOP.ANALYSED.AIR:LAMPOWPP_OSP'),
 ListData(shot=0, pass_=-1, signal_name='AIR_LAMPOWSOL', generic_name='', source_alias='air', type='Analysed', description='No longer written out', signal_status=1, mds_name='\\TOP.ANALYSED.AIR:LAMPOWSOL'),
 ListData(shot=0, pass_=-1, signal_name='AIR_LAMPOWSOL2', generic_name='', source_alias='air', type='Analysed', description='No longer written out', signal_status=1, mds_name='\\TOP.ANALYSED.AIR:LAMPOWSOL2'),
 ListData(shot=0, pass_=-1, signal_name='AIR_LAMPOWSOL_ISP', generic_name='', source_alias='air', type='Analysed', description='No longer written out', signal_status=1, mds_name='\\TOP.ANALYSED.AIR.LAMPOWSOL_:ISP'),
 ListData(shot=0, pass_=-1, signal_name='AIR_LAMPOWSOL_OSP', generic_name='', source_alias='air', type='Analysed', description='No longer written out', signal_status=1, mds_name='\\TOP.ANALYSED.AIR.LAMPOWSOL_:OSP'),
 ListData(shot=0, pass_=-1, signal_name='AIR_MINPOWER_DENSITY', generic_name='', source_alias='air', type='Analysed', description='Replaced', signal_status=1, mds_name='\\TOP.ANALYSED.AIR.MINPOWER:DENSITY'),
 ListData(shot=0, pass_=-1, signal_name='AIR_MINPOWER_DENSITY_30K', generic_name='', source_alias='air', type='Analysed', description='No longer used', signal_status=1, mds_name='\\TOP.ANALYSED.AIR.MINPOWER:DENSITY_30K'),
 ListData(shot=0, pass_=-1, signal_name='AIR_MINPOWER_DENSITY_ISP', generic_name='', source_alias='air', type='Analysed', description='Minimum heat flux to the divertor inner', signal_status=1, mds_name='\\TOP.ANALYSED.AIR.MINPOWER:DENSITY_ISP'),
 ListData(shot=0, pass_=-1, signal_name='AIR_MINPOWER_DENSITY_OSP', generic_name='', source_alias='air', type='Analysed', description='Minimum heat flux to the divertor outer', signal_status=1, mds_name='\\TOP.ANALYSED.AIR.MINPOWER:DENSITY_OSP'),
 ListData(shot=0, pass_=-1, signal_name='AIR_MINPOW_DENS_ISP_ELM', generic_name='', source_alias='air', type='Analysed', description='Minimum heat flux to the divertor inner for ELM', signal_status=1, mds_name='\\TOP.ANALYSED.AIR.MINPOW_DENS:ISP_ELM'),
 ListData(shot=0, pass_=-1, signal_name='AIR_MINPOW_DENS_OSP_ELM', generic_name='', source_alias='air', type='Analysed', description='Minimum heat flux to the divertor outer for ELM', signal_status=1, mds_name='\\TOP.ANALYSED.AIR.MINPOW_DENS:OSP_ELM'),
 ListData(shot=0, pass_=-1, signal_name='AIR_PASSNUMBER', generic_name='', source_alias='air', type='Analysed', description='', signal_status=1, mds_name='\\TOP.ANALYSED.AIR:PASSNUMBER'),
 ListData(shot=0, pass_=-1, signal_name='AIR_PEAKPOWER_DENSITY', generic_name='', source_alias='air', type='Analysed', description='No longer used', signal_status=1, mds_name='\\TOP.ANALYSED.AIR.PEAKPOWER:DENSITY'),
 ListData(shot=0, pass_=-1, signal_name='AIR_PEAKPOWER_DENSITY_30', generic_name='', source_alias='air', type='Analysed', description='No longer used', signal_status=1, mds_name='\\TOP.ANALYSED.AIR.PEAKPOWER:DENSITY_30'),
 ListData(shot=0, pass_=-1, signal_name='AIR_PEAKPOWER_POS', generic_name='', source_alias='air', type='Analysed', description='Replaced', signal_status=1, mds_name='\\TOP.ANALYSED.AIR.PEAKPOWER:POS'),
 ListData(shot=0, pass_=-1, signal_name='AIR_PEAKPOWER_POS_ISP', generic_name='', source_alias='air', type='Analysed', description='Radius of peak heat flux inner strike point', signal_status=1, mds_name=''),
 ListData(shot=0, pass_=-1, signal_name='AIR_PEAKPOWER_POS_OSP', generic_name='', source_alias='air', type='Analysed', description='Radius of peak heat flux outer strike point', signal_status=1, mds_name='\\TOP.ANALYSED.AIR.PEAKPOWER:POS_OSP'),
 ListData(shot=0, pass_=-1, signal_name='AIR_PHI_EXTENT', generic_name='', source_alias='air', type='Analysed', description='Replaced', signal_status=1, mds_name='\\TOP.ANALYSED.AIR:PHI_EXTENT'),
 ListData(shot=0, pass_=-1, signal_name='AIR_PHI_EXTENT_ISP', generic_name='', source_alias='air', type='Analysed', description='Toroidal extent of analysis path inner strike point', signal_status=1, mds_name='\\TOP.ANALYSED.AIR.PHI_EXTENT_:ISP'),
 ListData(shot=0, pass_=-1, signal_name='AIR_PHI_EXTENT_OSP', generic_name='', source_alias='air', type='Analysed', description='Toroidal extent of analysis path outer strike point', signal_status=1, mds_name='\\TOP.ANALYSED.AIR.PHI_EXTENT_:OSP'),
    """
    pass
    raise NotImplementedError

if __name__ == '__main__':
    pass