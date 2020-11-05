import numpy as np
import os
from scipy.interpolate import interp1d
"""
Get the S coordinate for given R,Z coordinates.

INPUTS:  

R (m)
Z (m)

OPTIONAL INPUTS:

surface_tol     How far the R,Z can be away from the tile surface (in metres) to still be assigned an S coordinate.
wall            'ILW' or 'C'
fast            Whether to use a faster method for getting the s coordinate. The faster method is accurate if input R,Z are further from the wall
plot            Whether to open a plot showing the input R,Z on a poloidal cross-section of JET.


OUTPUTS:

s               s coordinate in mm

"""


def get_s_coord(R,Z,surface_tol=0.02,wall='ILW',fast=False,plot=False):
    
    if np.array(R).shape == ():
        R = np.array([R])
        Z = np.array([Z])
    else:
        R = np.array(R)
        Z = np.array(Z)
        
    old_err_settings = np.seterr(all='ignore')

    # Load the S coordinate defition.
    s,sR,sZ = get_s_definition(wall)

    out_shape = R.shape
    R = np.array(R).flatten()
    Z = np.array(Z).flatten()
    s_out = np.zeros(R.shape)

    if fast:
        for i in range(R.size):
        	
            # Identify the nearest point
            dist = np.sqrt((sR - R[i])**2 + (sZ - Z[i])**2)
			            
            inds = np.argsort(dist)

            l = -1.
            j = 0            
            while l < 0:
                ind0 = inds[j]
                s0 = s[ind0]
                d0 = np.sqrt( (sR[ind0] - R[i])**2 + (sZ[ind0] - Z[i])**2 ) *1e3
                d1 = np.sqrt( (sR[ind0+1] - R[i])**2 + (sZ[ind0+1] - Z[i])**2 ) *1e3
                s1 = s[ind0+1]
                ds = s1 - s0                
                l = d0 * ( d0**2 + ds**2 - d1**2 ) / (2 * d0 * ds)
           
                j += 1
				
            
            dp = 1e3*np.sqrt( (sR[ind0+1] - sR[ind0])**2 +(sZ[ind0+1] - sZ[ind0])**2 )
            if abs(dp) > abs(ds):
                ds = np.sign(ds) * dp
            
            # How far we are along from s0 to s1
            l = d0 * ( d0**2 + ds**2 - d1**2 ) / (2 * d0 * ds)
            h = d0 * np.sqrt( 1 - ( ( d0**2 + ds**2 - d1**2 ) / (2 * d0 * ds) )**2 )

            if h/1e3 < surface_tol:
                s_out[i] = s0 + l

    else:
        for i in range(R.size):
            # Identify the nearest and second nearest points
            dist = np.sqrt((sR - R[i])**2 + (sZ - Z[i])**2)
            
            inds = np.argsort(dist)
    
            s_ = []
            dot_prod = []
            h = []
            for dind in range(max(-3,-inds[0]),min(4,s.size-inds[0])):
                d0 = dist[(inds[0]+dind) % dist.size]*1e3
                s0 = s[inds[0]+dind]
                d1 = dist[(inds[1]+dind) % dist.size]*1e3
                s1 = s[(inds[1]+dind) % dist.size]
                ds = s1 - s0
                dp = 1e3*np.sqrt( (sR[inds[1]] - sR[inds[0]])**2 +(sZ[inds[1]] - sZ[inds[0]])**2 )
                if abs(dp) > abs(ds):
                    ds = np.sign(ds) * dp
                    
                # How far we are along from s0 to s1
                l = d0 * ( d0**2 + ds**2 - d1**2 ) / (2 * d0 * ds)
                h.append( d0 * np.sqrt( 1 - ( ( d0**2 + ds**2 - d1**2 ) / (2 * d0 * ds) )**2 ) )
                
                if not np.isnan(l):
                    s_.append( s0 + l )
                    
                    rz_rev = get_R_Z(s_[-1])
                    
                    dir_along_surf = np.array([ sR[ (inds[1]+dind) % sR.size] - sR[(inds[0]+dind) % sR.size] , sZ[(inds[1]+dind) % sZ.size] - sZ[(inds[0]+dind) % sZ.size] ])
                    dir_to_wall = np.array([R[i] - rz_rev[0][0] ,Z[i] - rz_rev[1][0] ])
                    dir_along_surf = dir_along_surf / np.sqrt(np.sum(dir_along_surf**2) )
                    dir_to_wall = dir_to_wall / np.sqrt(np.sum(dir_to_wall**2) )
                    dot_prod.append( np.abs(np.dot(dir_to_wall,dir_along_surf)) )

            corr_ind = np.argmin(dot_prod)
            if h[corr_ind]/1.e3 < surface_tol:
                s_out[i] = s_[corr_ind]
            else:
                s_out[i] = -1


    s_out = np.reshape(s_out,out_shape)
    np.seterr(**old_err_settings)
    
    
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(sR,sZ,'k')
        plt.plot(R,Z,'ro')
        plt.axes().set_aspect('equal')

        for point_ind in range(R.size):
            if s_out[point_ind] > -1:
                rz_wall = get_R_Z(s_out[point_ind])
                plt.plot([R[point_ind], rz_wall[0][point_ind]],[Z[point_ind], rz_wall[1][point_ind]],'r--')
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        plt.show()
      
    return s_out
    
    
# Get R, Z corresponding to given s coordinate.
def get_R_Z(s_in,wall='ILW'):

    if np.array(s_in).shape == ():
        s_in = np.array([s_in])
    else:
        s_in = np.array(s_in)

    # Load the S coordinate defition.
    s,sR,sZ = get_s_definition(wall)
    sR = interp1d(s,sR,bounds_error=False)
    sZ = interp1d(s,sZ,bounds_error=False)

    out_shape = s_in.shape

    s_in = np.array(s_in).flatten()
    R_out = np.zeros(s_in.shape)
    Z_out = np.zeros(s_in.shape)


    R_out = sR(s_in)
    Z_out = sZ(s_in)
    R_out = np.reshape(R_out,out_shape)
    Z_out = np.reshape(Z_out,out_shape)

    return R_out,Z_out



'''
Get S coordinate definition.

Optional input: wall name: 'ILW' or 'C'

Returns 3 column vectors: s (mm), R(m), Z(m)
'''
def get_s_definition(wall='ILW'):
	
    if wall == 'ILW':
        s_def = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)),'s_coord_ILW_full.txt'))
    elif wall == 'C':
        s_def = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)),'sCoord_C.csv'),delimiter=',')
    
    return s_def[:,0],s_def[:,1],s_def[:,2]
    
    
def get_tile(s):
    tiles = np.zeros(s.shape)-1
    tiles[(s >= 162) & (s <= 414.9)] = 1
    tiles[(s >= 430.2) & (s <= 608.5)] = 3
    tiles[(s >= 711.6) & (s <= 925.5)] = 4
    tiles[(s >= 1061.8) & (s < 1125.5)] = 5.1
    tiles[(s >= 1125.5) & (s < 1190.1)] = 5.2
    tiles[(s >= 1190.1) & (s < 1254.3)] = 5.3
    tiles[(s >= 1254.3) & (s <= 1319)] = 5.4
    tiles[(s >= 1363.3) & (s <= 1552.6)] = 6
    tiles[(s >= 1622.2) & (s <= 1838.5)] = 7
    tiles[(s >= 1854.0) & (s <= 2133.9)] = 8
    return tiles