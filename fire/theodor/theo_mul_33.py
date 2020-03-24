# -*- coding: utf-8 -*-
import datetime
import sys
import os
from time import time as curTime
from time import asctime
import platform
from pathlib import Path

from numpy import sqrt,zeros,float32,arange,abs

from .congrid import congrid2d
from .heatpotential import heatpotential
from .diffstar import diffstar
from .weight import weight
from .klg_fit import klg_fit
from .err_msg import err_msg
from .calc_par import calc_par

try:
    import ctypes
    #libtheofast=ctypes.cdll.LoadLibrary('/home/lsartran/THEODOR/theofast/libtheofast.so')
    file_path = Path(__file__).absolute().parent  # os.path.dirname(os.path.abspath(__file__))
    if 'x86_64' in platform.machine():
        file_path = file_path / 'theofast/libtheofast_x64.so'
    else:
        # file_path = file_path / 'theofast/libtheofast_x32.so'
        file_path = file_path / 'theofast/libtheofast_x64.so'
    libtheofast = ctypes.cdll.LoadLibrary(str(file_path))
except:
    # Since non-theofast doesn't work anyway, just raise the exception.
    raise ImportError('Could not load THEOFAST shared object: {:s}'.format(Error))




def theo_mul_33(data, time, location, d_target, \
    alpha_bot, alpha_top_org, diff, lam, aniso, \
    lun=sys.stdout, test=None, ti_pr=None, x_Tb=None, y_Tb=None, \
    anim=None, a_frames=None, a_skip=None, a_fn=None, \
    dstar=0.35, foll=None, micro_t=None,verbose=1):
    #!Except=0    ; suppresses Math error output
    dtarget=d_target
    nr_tr_i=100 # number of depth indicees for t_profiles (to be independent on aux)
    # check the input array
    # structure of the input array = arr(y,time) ?
    nd_x, nd_t = data.shape
    n_time = time.size
    a_s = location.size
    if nd_t!=n_time:
        data = data.T
        rot = 1
        nd_x, nd_t = data.shape
    else:
        rot = 0

    if nd_t!=n_time:
        err_msg(9,halt=True)
    if nd_x!=a_s:
        location=arange(nd_x, dtype=float32)
        print('[THEODOR]            Mismatch between location length and data size !!!', file=lun)
        a_s = nd_x
    auy = data.shape[0]
    # define the output array
    qflux = zeros((auy,n_time), dtype=float32)
    # top edge condition temperature dependent ?
    alpha_slope = 0.0 # default, no T dependence
    alpha_tmp = alpha_top_org # LS
    if alpha_top_org.ndim==0 or alpha_top_org.size==a_s:
        alpha_top_0 = zeros(a_s, dtype=float32)+alpha_top_org
    else:
    # Extract temperature parameters
        # LS : should not be needed
        #if len(alpha_top_org.shape) == 2:
        #  alpha_tmp=alpha_top_org[:,0]
        #else:
        #  alpha_tmp=alpha_top_org
        # SQRT like ?
        if alpha_tmp.size==2:
            alpha_slope = (alpha_top_org[1]-alpha_top_org[0])/sqrt(1000.0)
            alpha_top_0 = zeros(a_s, dtype=float32)+alpha_top_org[0]
            if verbose > 4:
                print('[THEODOR]            SQRT like, slope: ',alpha_slope[0], file=lun)
        else:
            # KLG like
            if alpha_tmp.size==3:
                if alpha_top_org.size==3:
                    error, (aa0, alpha_slope, ta0, t_maxd) = klg_fit(alpha_top_org)
                    if error==3:
                        alpha_slope = 0.0 # error=3 - linear dependence
                    alpha_top_0 = zeros(a_s, dtype=float32)+alpha_top_org(0)
                else:
                    alpha_slope = zeros(a_s, dtype=float32)
                    aa0 = alpha_slope
                    ta0 = alpha_slope
                    for i in range(a_s):
                        error, (aatmp, alpha_slopetmp, ta0tmp, t_maxd) = klg_fit(alpha_top_org[:,i])
                        if error==3:
                            alpha_slope[0] = 0.0
                        else:
                            aa0[i] = aatmp
                            alpha_slope[i] = alpha_slopetmp
                            ta0[i] = ta0tmp
                    alpha_top_0 = alpha_top_org[0,:]
                if verbose > 4:
                    print('[THEODOR]            alpha_top fit:  Y(T)=a+b/(1+T/T0)**2,', file=lun)
                    print('[THEODOR]            a, b, T0: ',aa0(0), alpha_slope(0), ta0(0), file=lun)
            else:
                alpha_top_0 = zeros(a_s, dtype=float32)+alpha_top_org[0]
    alpha_top = alpha_top_0
    # averaged surface mesh width
    deltay = abs((location[1:a_s]-location[0:a_s-1]).mean())
    # fit parameters for diff and lam calculation
    # Y(T)=a+b/(1+T/T0)**2
    #   [lambda] = W/m/K  [Diff] = m**2/s
    error, (ad0, bd0, td0, t_maxd) = klg_fit(diff)
    if error>0: 
        err_msg(error)
    if error!=0 and error!=4:
        raise Exception('stop')
    error, (ac0, bc0, tc0, t_maxc) = klg_fit(lam)
    if error>0:
        err_msg(error)
    if error!=0 and error!=4:
        raise Exception('stop')
    plusac=1 if ac0 >= 0 else 0 # upper temperature limit if ac < 0
    plusad=1 if ad0 >= 0 else 0
    # normalized heat transmission
    #;alpha_bot/lam(0) ; 1/m
    #;alpha_top/lam(0)
    #; if 1./mean(alpha_top/lam(0)) ge 0.005 then begin
    #; print >>lun,' top heat transmission to low: (lam(0)/alpha_top) > 0.005 m'
    #; stop
    #; end
    cflux0 = 2*ac0*tc0
    tratio = tc0/td0
    cc = (1.0+bc0/ac0)/2.0
    htrans_b = 2.0*cc*alpha_bot/lam[0]
    htrans_f = 2.0*cc*alpha_top_0/lam[0]
    T_unit = 'C'
    if verbose > 4:
        print('[THEODOR]            Temperature unit: ',T_unit, file=lun)
    offset = 0 if T_unit == 'C' else -273.
    # get the time step and optimum mesh width
    # the integration happens between j-1 and j
    # the integration time is not changed in the first two time slices
    dt = arange(n_time, dtype=float32)
    #dt(1:n_time-1)=time(1:n_time-1)-time(0:n_time-2)
    #dt(0)=dt(1)
    dt[0:n_time-1] = time[1:n_time]-time[0:n_time-1]
    dt[n_time-1] = dt[n_time-2]
    (ad, bd, aux, delta, relwidth, cflux, htstar_b, htstar_f, factor, dstar, micro_t) = calc_par( dt[0], diff[0], aniso,
            ad0, bd0, dtarget, deltay, cflux0, htrans_b, htrans_f, dstar, \
            lun=lun, micro_t=micro_t,verbose=verbose)
    ay1 = auy-1
    ay2 = auy-2
    ay3 = auy-3
    ax1 = aux-1
    ax2 = aux-2
    ax3 = aux-3
    # Initializing the 2-D field of heat potential "h_pot" by assuming
    # that the initial x-dependence (depth) of heat potential can be described
    # by a quadratic parabola between top and bottom, where only for the latter
    # the heat potential is assumed to be constant laterally and is estimated
    # by the mean value for the surface of each tile, excluding the very edges
    h_pot = zeros((auy,aux), dtype=float32)
    top = heatpotential(data[1:auy-1,0],offset, tc0, cc)
    hpc = top.mean()  # averaged heat potential at the surface
    x=0.0
    for j in range(ax1+1):
        h_pot[1:ay2+1,j] = top+(hpc-top)*x**2
        x += relwidth
    # no losses at the edge
    h_pot[0,:] = h_pot[1,:]
    h_pot[ay1,:] = h_pot[ay2,:]
    s_start = datetime.datetime.now()
    if verbose > 2:
        print('[THEODOR]            Starting calculations at: ',s_start.ctime(), file=lun)
    i = 0 # used for T monitoring
    # *****************************************************************************
    # big time loop
    #print('[THEODOR]            test: ', cc, offset, ad, bd, tratio, tc0, td0, plusac)
    del_pot = zeros(h_pot.shape, dtype=float32)
    timeTheoFast = 0.0
    for j in range(1, n_time):
        dummy = data[:,j-1:j+1]
        m_data = congrid2d(dummy,a_s,micro_t+1)

        # if micro_t gt 1 and j mod 50 eq 0 then stop
        for k in range(1, micro_t+1):
            # alpha top can be T dependent
            if alpha_tmp.size==2 and alpha_slope[0]>0:
                alpha_top = alpha_slope*sqrt(abs(m_data[:,k-1]))+alpha_top_0
                htstar_f = 2.0*cc*alpha_top/lam[0]*delta
            if alpha_tmp.size==3 and alpha_slope[0]!=0:
                alpha_top = aa0+alpha_slope/(1.0+m_data[:,k-1]/Ta0)**2
                htstar_f = 2.0*cc*alpha_top/lam[0]*delta
            # LS: try to use libtheofast, C version of the following inner loop (x2 speedup)
            if libtheofast is not None:
                theoFastStart = curTime()
                print('del_pot.ctypes.data', del_pot.ctypes.data)
                libtheofast.heat_potential_time_step(ctypes.c_int(aux),ctypes.c_int(auy),ctypes.c_float(factor),
                                                     h_pot.ctypes.data, ctypes.c_float(cc), ctypes.c_int(plusac),
                                                     ctypes.c_float(tratio), ctypes.c_float(ad), ctypes.c_float(bd),
                                                     del_pot.ctypes.data)
                timeTheoFast += curTime() - theoFastStart
                h_pot += del_pot
            else:
                del_pot[:,:] = (h_pot[1:ay2+1,0:ax3+1] \
                    - 2*h_pot[1:ay2+1,1:ax2+1] \
                    + h_pot[1:ay2+1,2:ax1+1] \
                    + (h_pot[0:ay3+1,1:ax2+1] \
                    - 2*h_pot[1:ay2+1,1:ax2+1] \
                    + h_pot[2:ay1+1,1:ax2+1]) * factor) \
                    *diffstar(h_pot[1:ay2+1,1:ax2+1], cc, plusac, tratio, ad, bd)
                hpleft = (h_pot[0,0:ax3+1] \
                    - 2*h_pot[0,1:ax2+1] \
                    + h_pot[0,2:ax1+1] \
                    + (h_pot[1,1:ax2+1] \
                    - h_pot[0,1:ax2+1]) * 2*factor) \
                    *diffstar(h_pot[0,1:ax2+1], cc, plusac, tratio, ad, bd)
                hpright= (h_pot[ay1,0:ax3+1] \
                    - 2*h_pot[ay1,1:ax2+1] \
                    + h_pot[ay1,2:ax1+1] \
                    + (h_pot[ay2,1:ax2+1] \
                    - h_pot[ay1,1:ax2+1]) * 2*factor) \
                    *diffstar(h_pot[ay1,1:ax2+1], cc, plusac, tratio, ad, bd)
                # the heat potential for this time step
                h_pot[1:ay2+1,1:ax2+1] = h_pot[1:ay2+1,1:ax2+1]+del_pot
                h_pot[0,1:ax2+1] = h_pot[0,1:ax2+1] + hpleft
                h_pot[ay1,1:ax2+1] = h_pot[ay1,1:ax2+1] + hpright
            # next surface temperature to the top
            h_pot[:,0] = heatpotential(m_data[:,k],offset, tc0, cc)
            # bottom corners (are handeled by the edge already
            #; h_pot(0,ax2)=h_pot(1,ax2)
            #; h_pot(ay1,ax2)=h_pot(ay2,ax2)
            # bottom
            h_pot[0:ay1+1,ax1] = hpc + (4*h_pot[0:ay1+1,ax2]-h_pot[0:ay1+1,ax3]-3*hpc) \
                * weight((hpc+h_pot[0:ay1+1,ax1])/2.,htstar_b,cc,plusac)
            # front
            t_aux = h_pot[1:ay2+1,0]
            h_pot[1:ay2+1,0] = t_aux + (4*h_pot[1:ay2+1,1]-h_pot[1:ay2+1,2]-3*t_aux) \
                * weight(h_pot[1:ay2+1,1]+(t_aux-h_pot[1:ay2+1,2])/2.,htstar_f[1:ay2+1],cc,plusac)
            # top corners
            h_pot[0,0] = h_pot[1,0]-0.333333*(4*(h_pot[1,1]-h_pot[0,1]) \
                - (h_pot[1,2]-h_pot[0,2]))
            h_pot[ay1,0] = h_pot[ay2,0]-0.333333*(4*(h_pot[ay2,1]-h_pot[ay1,1]) \
                - (h_pot[ay2,2]-h_pot[ay1,2]))
            # end of micro_time step loop
        # surface heat flux
        qflux[0:ay1+1,j] = cflux*(3.*h_pot[0:ay1+1,0]-4*h_pot[0:ay1+1,1]+h_pot[0:ay1+1,2])
        # this is the handling for changing time vector
#		if dt[j]<=0.95*dt[j-1] or dt[j]>=1.05*dt[j-1]:
#			print >>lun,'change of time increment: ',dt[j-1],dt[j]
#			print >>lun,'@ time:                   ',time[j]
#			# What are the new parameters (mesh size)
#			# common block dimless
#			(ad, bd, aux, delta, relwidth, cflux, htstar_b, htstar_f, factor, dstar, micro_t) = \
#				calc_par( \
#					dt[j],diff[0],aniso, \
#					ad0, bd0, dtarget, deltay, cflux0, htrans_b, htrans_f, dstar, \
#					test=test, lun=lun, micro_t=micro_t)
#			ax1 = aux-1
#			ax2 = aux-2
#			ax3 = aux-3
#			# change the heat potential array according the new mesh
#			# h_pot(surface,depth)=h_pot(y,x)=h_pot(auy,aux)
#			h_pot = congrid2d(h_pot,auy,aux)
#			del_pot = congrid2d(del_pot,auy,aux)
    # end of big time step loop
    s_end = datetime.datetime.now()
    td = s_end-s_start
    if verbose > 2:
        print('[THEODOR]            Finished calculations at: ',s_end.ctime(), file=lun)
        print('[THEODOR]            Elapsed time: {:.0f}s (THEOFast: {:.0f}s)'.format(td.seconds,timeTheoFast), file=lun)
    extra_results = {}
    if rot==1: 
        qflux = qflux.T
    return qflux, extra_results

