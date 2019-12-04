from numpy import sqrt

def calc_par(dt,diff,aniso,ad0, bd0, dtarget, deltay, cflux0, htrans_b, htrans_f, dstar, lun=None, micro_t=None, flu=None,verbose=1):
    # INPUT ad0, bd0, dtarget, deltay, cflux0, htrans_b, htrans_f, dstar
    # OUTPUT ad, bd, aux, delta, relwidth, cflux, htstar_b, htstar_f, factor, dstar
    if not micro_t: 
        micro_t = 1
    # maximum number of meshes in depth
    max_meshes = 1500
    # minimum number of meshes required in depth
    min_meshes = 5
    #; multiply = [6,10] ; not implemented yet
    # calculates the dimensionless parameters for calculation
    #; local: aux, relwidth
    #tile dependent parameter
    micro_loop = 0
    flag1 = True
    flag2 = True
    while flag1 and flag2:
        dx = sqrt(diff*dt/micro_t/dstar) # optimum mesh width
        aux = int(dtarget/dx+0.5) # mesh number
        if aux>max_meshes:   # adjust the mesh numbers
            dstar = dstar*(max_meshes*dx/dtarget)**2
            if verbose > 5:
                print('[THEODOR]            dstar changed to: ',dstar, file=lun)
        else:
            flag1 = False
        if flag1==False and aux<=min_meshes:
            if verbose > 5:
                print('[THEODOR]            Mesh number: ',aux, file=lun)
                print('[THEODOR]            Time increment: ',dt, file=lun)
            micro_t = int(2.0*min_meshes/aux)+1
            micro_loop += 1
            if verbose > 5:
                print('[THEODOR]            New micro_steps: ',micro_t, file=lun)
            if micro_loop<=5:
                pass # LS will loop again # required time resolution to
            else:
                flag2 = False
            err_msg(8, halt=True)
    relwidth = 1.0/aux
    delta = dtarget*relwidth  # elementary mesh width in x direction
    factor = aniso*(delta/deltay)**2
    star = dt/micro_t/delta**2
    if factor>1.0:
        print('[THEODOR]                 WARNING !!!', file=lun)
        print('[THEODOR]                 Ratio x/y mesh size => 1 - numerically unstable', file=lun)
        print('[THEODOR]                 Anisotropy set to 0.', file=lun)
        aniso = 0.0
        factor = 0.0
    # calculate dimenionless parameters
    ad = ad0*star
    bd = bd0*star
    cflux = cflux0/(delta+delta)
    # !!!!! changed by alh 13.01.2001 to make it consistent with modelling
    #   check the reason
    htstar_b = htrans_b*delta # *2
    htstar_f = htrans_f*delta # *2
    if verbose > 5:
        print('[THEODOR]            Mesh number:               ',aux, file=lun)
        print('[THEODOR]            Relwidth:                  ',relwidth, file=lun)
        print('[THEODOR]            Elementary mesh width:     ',delta, file=lun)
        print('[THEODOR]            Mesh width in y direction: ',deltay, file=lun)
        print('[THEODOR]            Time increment:            ',dt/micro_t, file=lun)
        print('[THEODOR]            Effective dstar:           ',diff*dt/micro_t/delta/delta, file=lun)
        print('[THEODOR]            Micro time steps:          ',micro_t, file=lun)
        # print >>lun,'adstar: ',ad
        # print >>lun,'bdstar: ',bd
        print('[THEODOR]            Factor:                    ',factor, file=lun)
        # print >>lun,'cflux:  ',cflux
    return (ad, bd, aux, delta, relwidth, cflux, htstar_b, htstar_f, factor, dstar, micro_t)

