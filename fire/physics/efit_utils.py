#!/usr/bin/env python

"""Tools for extracting efit information

Summary of external libraries for using efit data:
    - mastu_exhaust.read_efit() - Read efit data from uda, epm or efit out files
                                  efit_data = read_uda(shot)
                                  read_epm(filename, calc_bfield = True)
                                  read_efitout(filename, calc_bfield = True)
    - mastu_exhaust.mastu_equilibrium.mastu_equilibrium() - Inherits from pyEquilibrium, adds functionality
                                                            equil = mastu_equilibrium()
                                                            equil.load_efitpp('../data/epm044442.nc', time=0.5)
                                                            print(equil.get_baffle_clearance())
    - mastu_exhaust.divertor_geometry.divertor_geometry() - Class for (over)plotting and calculating equilibrium props
                                             div_geom = divertor_geometry(filename='../data/epm044442.nc', time=0.55)
                                             div_geom.plot(annotate = True, mark_oxpts = True)
                                             div_geom2 = divertor_geometry(filename='../data/epm044442.nc', time=0.65)
                                             div_geom.plot(equils=div_geom2)
    - fluxsurface_tracer.fsurface() - class used for tracing field lines and calculating intersection points etc
                        fsurface1, fsurface2 = trace_flux_surface(efit_data, i, start_r, start_z, cut_surface=True)
                        intr_x, intr_y = furface1.find_intersection(r_start, r_end, z_start, z_end)

Created: Tom Farley, Nov 2021
"""

import logging, datetime
from typing import Union, Iterable, Sequence, Tuple, Optional, Any, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline, interp1d,  Rbf, LinearNDInterpolator

from fire.interfaces.uda_utils import get_uda_client
from fire.misc.utils import make_iterable


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


def get_equilibrium_obj(shot, time, with_bfield=True, filename=None):
    """Efit reading method used by JRH"""
    machine = 'MASTU'
    try:
        from mastu_exhaust_analysis.mastu_equilibrium import mastu_equilibrium
        if filename is not None:
            equilibrium = mastu_equilibrium(gfile=filename, time=time, device=machine,
                                            with_bfield=with_bfield)
        elif shot < 44849:
            path_uda_scratch = '/common/uda-scratch/lkogan/efitpp_eshed/epm0' + str(shot) + '.nc'
            equilibrium = mastu_equilibrium(efitpp_file=path_uda_scratch, time=time, device=machine,
                                            with_bfield=with_bfield)
        else:
            equilibrium = mastu_equilibrium(shot=shot, time=time, device=machine, with_bfield=with_bfield)
    except ImportError as e:
        raise e
    except Exception as e:
        raise e
    return equilibrium

def extract_eqm_info_for_path(shot, path_data, times, path_name='path0'):
    from pyEquilibrium.equilibrium import equilibrium

    path = path_name

    efit_path = get_efit_file_path(shot=shot)
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

def get_efit_file_path(shot):
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
            epm_fn = get_efit_file_path(shot)
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
    return all_time_sep_r, all_time_sep_z, r_fine,z_fine

def get_3d_field_interpolator(efit_data, field='psiN'):

    z_grid = efit_data[field]
    z_grid_transposed = np.swapaxes(z_grid, 1, 2)
    points = np.array(list(zip(ttt.ravel(), rrr.ravel(), zzz.ravel())))
    z_flat = z_grid_transposed.ravel()
    interp_linear_3d = LinearNDInterpolator(points, z_flat)

    return interp_linear_3d

def get_2d_spline_interpolator(efit_data, tind, field='psiN'):
    data = efit_data[field][tind, :, :]
    interp_t = RectBivariateSpline(efit_data['r'], efit_data['z'], np.transpose(data))
    return interp_t

def get_nearest_efit_t(efit_data, t):

    t_efit = efit_data['t']
    t_ind = np.argmin(np.abs(t_efit-t))
    t_nearest = t_efit[t_ind]

    return t_ind, t_nearest

def interp_3d_field_loop(efit_data, r, z, times, field='psiN'):

    # data_grid = efit_data[field]

    out = np.zeros((len(times), len(r), len(z)))

    data_t_unique = []
    t_unique = []

    # get values at closest unique efit times at given r and z
    t_ind_prev = -1
    for i_t, t in enumerate(times):
        t_ind, t_efit = get_nearest_efit_t(efit_data, t)
        if t_ind == t_ind_prev:
            continue
        interp_t = get_2d_spline_interpolator(efit_data, t_ind, field=field)
        data_t = interp_t(r, z)

        data_t_unique.append(data_t)
        t_unique.append(t_efit)

    data_t_unique = np.array(data_t_unique)
    shape = data_t_unique.shape

    # Interpolate to actual requested times
    for i_r, r in enumerate(efit_data['r']):
        for i_z, z in enumerate(efit_data['z']):
            data_at_point = data_t_unique[:, i_r, i_z]
            out[:, i_r, i_z] = interp1d(t_unique, data_at_point)

    return out

def interp_3d_field_direct(efit_data, r, z, times, field='psiN'):

    f_interp = get_3d_field_interpolator(efit_data, field=field)
    out = f_interp(list(zip(times, r, z)))

    return out




if __name__ == '__main__':
    shot = 44345
    time = 0.5

    efit_data = get_efit_data(shot=shot, calc_bfield=True)
    # Mask off points in the EFIT++ grid that are outside the wall
    rr, zz = np.meshgrid(efit_data['r'], efit_data['z'])
    ttt, rrr, zzz = np.meshgrid(efit_data['t'], efit_data['r'], efit_data['z'])
    tindx = 1
    psiN_interp_t = RectBivariateSpline(efit_data['r'], efit_data['z'], np.transpose(efit_data['psiN'][tindx, :, :]))
    psiN = psiN_interp_t(1, 1)

    r = np.linspace(1, 1.3, 31)
    z = np.linspace(-1.3, -1.5, 21)
    t = np.linspace(0.05, 0.3, 26)

    psiNs_loop = interp_3d_field_loop(efit_data, r, z, t, field='psiN')

    psiN_interp = get_3d_field_interpolator(efit_data, field='psiN')
    psiNs_direct = interp_3d_field_direct(efit_data, field='psiN')

    psiN_grid = np.swapaxes(efit_data['psiN'], 1, 2)
    points = np.array(list(zip(ttt.ravel(), rrr.ravel(), zzz.ravel())))
    psiN_flat = psiN_grid.ravel()
    psiN_interp = LinearNDInterpolator(points, psiN_flat)
    psiNs = psiN_interp(list(zip(t, r, z)))

    equilibrium = get_equilibrium_obj(shot=shot, time=time, with_bfield=True)

    b = read_bfield_data(shot, r, z, t)

    print(b)
    pass