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
from fire.misc.utils import make_iterable

from pyEquilibrium.equilibrium import equilibrium
from mastu_exhaust_analysis import mastu_equilibrium

logger = logging.getLogger(__name__)
logger.propagate = False
logger.setLevel(logging.DEBUG)

def get_efit_data(shot, calc_bfield=False, filename=None):
    """Efit reading method used by JRH"""
    try:
        from mastu_exhaust_analysis.read_efit import read_uda, read_epm, read_efitout
        if filename is not None:
            efit_data = read_efitout(filename, calc_bfield=calc_bfield)
        elif shot < 44849:
            path_uda_scratch = '/common/uda-scratch/lkogan/efitpp_eshed/epm0' + str(shot) + '.nc'
            efit_data = read_epm(path_uda_scratch, calc_bfield=calc_bfield)
        else:
            efit_data = read_uda(shot, calc_bfield=calc_bfield)
    except ImportError as e:
        raise e
    except Exception as e:
        raise e
    return efit_data

def extract_eqm_info_for_path(shot, path_data, times, path_name='path0'):
    path = path_name

    efit_path = get_efit_path(shot=shot)
    efit_times = get_efit_converted_times(shot)

    in_frame_str = '_in_frame'
    coord_path = f'i{in_frame_str}_{path}'
    if coord_path not in path_data.coords:
        in_frame_str = ''
        coord_path = f'i{in_frame_str}_{path}'

    r, z = path_data['R'], path_data['z']
    psi_norm = []
    for t in efit_times:
        eqm_obj = equilibrium(shot=shot, device='MASTU', time=t, efitpp_file=efit_path)
        psi_norm.append(eqm_obj.psiN(r, z)[0])

    raise NotImplementedError

def get_efit_path(shot):
    import os
    path = None
    if os.path.isfile('/common/uda-scratch/shenders/efit/mastu/{}/efit_shenders_01/efitOut.nc'.format(shot)):
        path='/common/uda-scratch/shenders/efit/mastu/{}/efit_shenders_01/efitOut.nc'.format(shot)
    if os.path.isfile('/common/projects/codes/equilibrium/efit++/mast_data/runs_mastu/mastu/{}/efit_lkogan_01/efitOut.nc'.format(shot)):
        path='/common/projects/codes/equilibrium/efit++/mast_data/runs_mastu/mastu/{}/efit_lkogan_01/efitOut.nc'.format(shot)
    if os.path.isfile('/common/uda-scratch/lkogan/efitpp_eshed/efitOut_{}.nc'.format(shot)):
        path='/common/uda-scratch/lkogan/efitpp_eshed/efitOut_{}.nc'.format(shot)
    return path

def read_equilibrium_signal(shot, signal='/epm/equilibriumStatusInteger', machine='MASTU'):
    uda_module, client = get_uda_client()
    try:
        data = client.get(signal, shot)
    except Exception as e:
        if isinstance(shot, (int, float)):
            epm_fn = get_efit_path(shot)
        else:
            epm_fn = shot.format(shot=shot, pulse=shot)
        epm_times = client.get('/epm/time', epm_fn).data
        data = client.get('/epm/equilibriumStatusInteger', epm_fn)
    return data

def get_efit_converted_times(shot):
    converged_status = read_equilibrium_signal(shot, signal='/epm/equilibriumStatusInteger')
    epm_times = read_equilibrium_signal(shot, signal='/epm/time')

    mask_converged = converged_status.data == 1
    t_converged = epm_times[mask_converged]
    return t_converged, epm_times, mask_converged

def get_eqm_obj(shot, t, epm_path='/common/uda-scratch/lkogan/efitpp_eshed/epm0{shot}.nc'):
    t_converged, epm_times, mask_converged = get_efit_converted_times(shot=shot)
    if np.any(np.isclose(t, t_converged)):
        try:
            eqm_data = equilibrium(shot=shot, device='MASTU', time=t)
        except Exception as e:
            epm_path_fn = epm_path.format(shot=shot)
            eqm_data = equilibrium(shot=epm_path_fn, device='MASTU', time=t)

        # for i, (ri, zi) in enumerate(zip(r, z)):
        #     # Magnetic field data at this point in space
        #     br = eqm_data.BR(ri, zi)[0][0]
        #     bz = eqm_data.BZ(ri, zi)[0][0]
        #     bt = eqm_data.Bt(ri, zi)[0][0]
        #
        #     b.loc[dict(t=t, path=i, B_vec=['r', 'z', 'phi'])] = np.array([br, bz, bt])
        logger.debug(f'Read equilibrium data for t={t}')
    else:
        logger.warning(f'"{t}" not in efit converged times: {t_converged}')
        eqm_data = None

    return eqm_data

def read_bfield_data(shot, r, z, t, epm_path='/common/uda-scratch/lkogan/efitpp_eshed/epm0{shot}.nc'):
    t_converged, epm_times, mask_converged = get_efit_converted_times(shot=shot)

    print(f't requested: {t}')
    print(f't converged: {t_converged}')

    logger.info(f'Reading equilibrium data for shot {shot}')

    b = xr.DataArray(data=np.full((len(t), len(r), 3), np.nan), dims=('t', 'path', 'B_vec'),
                     coords={'t': t, 'path': np.arange(len(r)), 'B_vec': ['r', 'z', 'phi']}, name='B_field')

    t_start = datetime.datetime.now()
    for ti in t:
        if np.any(np.isclose(ti,t_converged)):
            try:
                eqm_data = equilibrium(shot=shot, device='MASTU', time=ti)
            except Exception as e:
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
    equil.load_efitpp(44860, time=0.5)

    # equil.load_efitpp('epm044427.nc', time=0.5)

def read_efit_reconstruction(shot_no, efit_reconstruction=None):
    timeout = 20 * 60  # 20 minutes
    while (efit_reconstruction is None) and (timeout > 0):
        try:
            EFIT_path_default = '/common/uda-scratch/lkogan/efitpp_eshed'
            efit_reconstruction = mclass(EFIT_path_default + '/epm0' + shot_no + '.nc',
                                         pulse_ID=shot_no)

        except:
            print('EFIT missing, waiting 20 seconds')
        tm.sleep(20)
        timeout -= 20

def efit_reconstruction_to_separatrix_on_foil(efit_reconstruction,refinement=1000):
    from scipy.signal import find_peaks, peak_prominences as get_proms
    from scipy.interpolate import interp1d, interp2d

    R_centre_column = 0.261  # m

    all_time_sep_r = []
    all_time_sep_z = []
    r_fine = np.unique(np.linspace(efit_reconstruction.R.min(),efit_reconstruction.R.max(),
                    refinement).tolist() + np.linspace(R_centre_column-0.01,R_centre_column+0.08,refinement).tolist())
    z_fine = np.linspace(efit_reconstruction.Z.min(),efit_reconstruction.Z.max(),refinement)
    interp1 = interp1d([1.1,1.5],[-1.5,-1.75],fill_value="extrapolate",bounds_error=False)
    interp2 = interp1d([1.1,1.5],[-1.5,-1.2],fill_value="extrapolate",bounds_error=False)
    for time in range(len(efit_reconstruction.time)):
        gna = efit_reconstruction.psidat[time]
        sep_up = efit_reconstruction.upper_xpoint_p[time]
        sep_low = efit_reconstruction.lower_xpoint_p[time]
        x_point_z_proximity = np.abs(np.nanmin([efit_reconstruction.upper_xpoint_z[time],
                            efit_reconstruction.lower_xpoint_z[time],-0.573-0.2]))-0.2	# -0.573 is an arbitrary treshold in case both are nan

        psi_interpolator = interp2d(efit_reconstruction.R,efit_reconstruction.Z,gna)
        psi = psi_interpolator(r_fine,z_fine)
        psi_up = -np.abs(psi-sep_up)
        psi_low = -np.abs(psi-sep_low)
        all_peaks_up = []
        all_z_up = []
        all_peaks_low = []
        all_z_low = []
        for i_z,z in enumerate(z_fine):
            # psi_z = psi[i_z]
            peaks = find_peaks(psi_up[i_z])[0]
            all_peaks_up.append(peaks)
            all_z_up.append([i_z]*len(peaks))
            peaks = find_peaks(psi_low[i_z])[0]
            all_peaks_low.append(peaks)
            all_z_low.append([i_z]*len(peaks))
        all_peaks_up = np.concatenate(all_peaks_up).astype(int)
        all_z_up = np.concatenate(all_z_up).astype(int)
        found_psi_up = np.abs(psi_up[all_z_up,all_peaks_up])
        all_peaks_low = np.concatenate(all_peaks_low).astype(int)
        all_z_low = np.concatenate(all_z_low).astype(int)
        found_psi_low = np.abs(psi_low[all_z_low,all_peaks_low])

        # plt.figure()
        # plt.plot(z_fine[all_z_up[found_psi_up<(gna.max()-gna.min())/500]],r_fine[all_peaks_up[found_psi_up<(gna.max()-gna.min())/500]],'+b')
        # plt.plot(z_fine[all_z_low[found_psi_low<(gna.max()-gna.min())/500]],r_fine[all_peaks_low[found_psi_low<(gna.max()-gna.min())/500]],'+r')
        all_peaks_up = all_peaks_up[found_psi_up<(gna.max()-gna.min())/500]
        all_z_up = all_z_up[found_psi_up<(gna.max()-gna.min())/500]
        all_peaks_low = all_peaks_low[found_psi_low<(gna.max()-gna.min())/500]
        all_z_low = all_z_low[found_psi_low<(gna.max()-gna.min())/500]

        # plt.figure()
        # plt.plot(z_fine[all_z_up],r_fine[all_peaks_up],'+b')
        # plt.plot(z_fine[all_z_low],r_fine[all_peaks_low],'+r')
        select = np.logical_or( interp1(r_fine[all_peaks_up])>z_fine[all_z_up] , interp2(r_fine[all_peaks_up])<z_fine[all_z_up] )
        all_peaks_up = all_peaks_up[select]
        all_z_up = all_z_up[select]
        select = np.logical_or( interp1(r_fine[all_peaks_low])>z_fine[all_z_low] , interp2(r_fine[all_peaks_low])<z_fine[all_z_low] )
        all_peaks_low = all_peaks_low[select]
        all_z_low = all_z_low[select]

        left_up = []
        right_up = []
        left_up_z = []
        right_up_z = []
        left_low = []
        right_low = []
        left_low_z = []
        right_low_z = []
        for i_z,z in enumerate(z_fine):
            if i_z in all_z_up:
                temp = all_peaks_up[all_z_up==i_z]
                if len(temp) == 1:
                    right_up.append(temp[0])
                    right_up_z.append(i_z)
                elif len(temp) == 2:
                    # # if r_fine[temp.min()]>R_centre_column or np.abs(z)<x_point_z_proximity:
                    # if r_fine[temp.min()]>R_centre_column_interpolator(-np.abs(z)):
                    left_up.append(temp.min())
                    left_up_z.append(i_z)
                    right_up.append(temp.max())
                    right_up_z.append(i_z)
                elif len(temp) == 3:
                    left_up.append(np.sort(temp)[1])
                    left_up_z.append(i_z)
                    right_up.append(temp.max())
                    right_up_z.append(i_z)
                elif len(temp) == 4:
                    left_up.append(np.sort(temp)[1])
                    left_up_z.append(i_z)
                    right_up.append(np.sort(temp)[2])
                    right_up_z.append(i_z)
            if i_z in all_z_low:
                temp = all_peaks_low[all_z_low==i_z]
                if len(temp) == 1:
                    right_low.append(temp[0])
                    right_low_z.append(i_z)
                elif len(temp) == 2:
                    # # if r_fine[temp.min()]>R_centre_column or np.abs(z)<x_point_z_proximity:
                    # if r_fine[temp.min()]>R_centre_column_interpolator(-np.abs(z)):
                    left_low.append(temp.min())
                    left_low_z.append(i_z)
                    right_low.append(temp.max())
                    right_low_z.append(i_z)
                elif len(temp) == 3:
                    left_low.append(np.sort(temp)[1])
                    left_low_z.append(i_z)
                    right_low.append(temp.max())
                    right_low_z.append(i_z)
                elif len(temp) == 4:
                    left_low.append(np.sort(temp)[1])
                    left_low_z.append(i_z)
                    right_low.append(np.sort(temp)[2])
                    right_low_z.append(i_z)
        # sep_r = [left_up,right_up,left_low,right_low]
        # sep_z = [left_up_z,right_up_z,left_low_z,right_low_z]
        all_time_sep_r.append([left_up,right_up,left_low,right_low])
        all_time_sep_z.append([left_up_z,right_up_z,left_low_z,right_low_z])
    return all_time_sep_r,all_time_sep_z,r_fine,z_fine

if __name__ == '__main__':
    shot = 44345
    r = np.linspace(1, 1.3, 31)
    z = np.linspace(-1.3, -1.5, 21)
    t = np.linspace(0.05, 0.3, 26)
    b = read_bfield_data(shot, r, z, t)

    print(b)
    pass