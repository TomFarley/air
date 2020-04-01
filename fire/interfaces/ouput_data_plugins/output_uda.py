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
    
    
    NETCDF output code from /home/athorn/IR/Latest/sched_air_netcdf/src/netcdfout.pro:
    	;write out the coordinates
	nt=n_elements(tout)
	nrv=n_elements(rout_norm)

	;dimensions
	rc=putdata(nt, stepId='dimension', group='/'+prefix, name='time')
	rc=putdata(nrv, stepId='dimension', group='/'+prefix, name='radius')
	rc=putdata(1, stepId='dimension', group='/'+prefix, name='value')
	;coordinates
	rc=putdata(tout, stepId='coordinate', group='/'+prefix, name='time', $
		units='t', label='Time', class='time')
	rc=putdata(rout_norm, stepId='coordinate', $
				group='/'+prefix, name='radius', $
				units='t', label='Time')
	rc=putdata([0], stepId='coordinate', group='/'+prefix, name='value', $
		units='', label='Value')

	;write out the data
	rc=putdata(qmin(sti:eti), stepid='variable', $
		group='/'+prefix, name='minpower_density', $
		dimensions='time', units='MW/m^2', label='Min q')

	rc=putdata(qmax(sti:eti), stepid='variable', $
		group='/'+prefix, name='pkpower_density', $
		dimensions='time', units='MW/m^2', label='Max q')
;
   if (nalpha gt 1) then begin
		rc=putdata(qmin_ELM(sti:eti), stepid='variable', $
			group='/'+prefix, name='minpower_density_elm', $
			dimensions='time', units='MW/m^2', label='Min q (ELM alpha)')

		rc=putdata(qmax_ELM(sti:eti), stepid='variable', $
			group='/'+prefix, name='pkpower_density_elm', $
			dimensions='time', units='MW/m^2', label='Max q (ELM alpha)')
   endif
;
	rc=putdata(qmax(sti:eti), stepid='variable', $
		group='/'+prefix, name='peakpower_pos', $
		dimensions='time', units='m', label='R(peak q)')

	rc=putdata(ptot(sti:eti), stepid='variable', $
		group='/'+prefix, name='ptot', $
		dimensions='time', units='MW', label='P(tot)')


	rc=putdata(qprofiles, stepid='variable', $
		group='/'+prefix, name='qprofile', $
		dimensions='radius, time', units='MW/m^2', label='q_profile')

	rc=putdata(tprofiles, stepid='variable', $
		group='/'+prefix, name='tprofile', $
		dimensions='radius, time', units='MW/m^2', label='t_profile')

   if (nalpha gt 1) then begin
		rc=putdata(qprofiles_elm, stepid='variable', $
			group='/'+prefix, name='qprofile_elm', $
			dimensions='radius, time', units='MW/m^2', label='q_profile_elm')
   endif

	rc=putdata(numsatpix, stepid='variable', $
		group='/'+prefix, name='satpixels', $
		units='', label='Num sat. pix.', dimensions='value')

;write out the radius
     
	rc=putdata(rout, stepid='variable', $
		group='/'+prefix, name='rcoord', $
		units='m', label='R coord', dimensions='radius')

; energies
;FL 24/11/2003: adding PTOT and ETOT trace
	rc=putdata(ptot(sti:eti), stepid='variable', $
		group='/'+prefix, name='ptot', $
		dimensions='time', units='MW', label='P(tot)')

	rc=putdata(1.e3*etot(sti:eti), stepid='variable', $
		group='/'+prefix, name='etot', $
		dimensions='time', units='kJ', label='E(tot)')

	rc=putdata(1.e3*etotsum(sti:eti), stepid='variable', $
		group='/'+prefix, name='etotsum', $
		dimensions='time', units='kJ', label='E_sum(tot)')

    if (nalpha gt 1) then begin

	rc=putdata(ptot_elm(sti:eti), stepid='variable', $
		group='/'+prefix, name='ptot_elm', $
		dimensions='time', units='MW', label='P(tot) ELM')

	rc=putdata(1.e3*etot_elm(sti:eti), stepid='variable', $
		group='/'+prefix, name='etot_elm', $
		dimensions='time', units='kJ', label='E(tot) ELM')

	rc=putdata(1.e3*etotsum_elm(sti:eti), stepid='variable', $
		group='/'+prefix, name='etotsum_elm', $
		dimensions='time', units='kJ', label='E_sum(tot) ELM')

   endif
    
; Some Info about the TEMPERATURE
; time evolution of maxima/minima
   maxtemp=fltarr(max_times)
   for i=0l,max_times-1 do begin
      maxtemp(i)=max(tprofiles(*,i))
   endfor

	rc=putdata(maxtemp(sti:eti), stepid='variable', $
		group='/'+prefix, name='temperature', $
		dimensions='time', units='Degrees C', label='Max temp.')

;  write out the configuration
	rc=putdata(s, stepid='variable', $
		group='/'+prefix, name='camera_view', dimensions='value')

;  write out the value of alpha used
	rc=putdata([alphaconst[0]], stepid='variable', $
			group='/'+prefix, name='alphaconst', $
			units='kW.m^-2', label='Alpha', dimensions='value')

  if (nalpha gt 1) then begin
		rc=putdata([alphaconst[1]], stepid='variable', $
			group='/'+prefix, name='alphaconst_elm', $
			units='kW.m^-2', label='Alpha ELM', dimensions='value')
   endif

;  write out the rstart         
	rc=putdata([r_start], stepid='variable', $
		group='/'+prefix, name='r_start', $
		units='m', label='r_start', dimensions='value')

;  write out the phistart         
	rc=putdata([phi_start], stepid='variable', $
		group='/'+prefix, name='phi_start', $
		units='degrees', label='phi_start', dimensions='value')

;  write out the zstart         
	rc=putdata([z_start], stepid='variable', $
		group='/'+prefix, name='z_start', $
		units='m', label='z_start', dimensions='value')

;  write out the rextent        
	rc=putdata([r_extent], stepid='variable', $
		group='/'+prefix, name='r_extent', $
		units='m', label='r_extent', dimensions='value')

	rc=putdata([phi_extent], stepid='variable', $
		group='/'+prefix, name='phi_extent', $
		units='degrees', label='phi_extent', dimensions='value')

	rc=putdata([z_extent], stepid='variable', $
		group='/'+prefix, name='z_extent', $
		units='degrees', label='z_extent', dimensions='value')

    """
    pass
    raise NotImplementedError

if __name__ == '__main__':
    pass