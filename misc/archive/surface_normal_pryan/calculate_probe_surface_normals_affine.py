import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import pyuda
client=pyuda.Client()



def RZ_shifts(R_unshifted,Z_unshifted,tile,upper_or_lower):
    aff_trans = client.geometry("/affinetransforms", 50000, no_cal=True)
    centrecolumn_tiles_mat=aff_trans.data['asbuilt/rz_2d/centrecolumn_tiles/data'].matrix
    centrecolumn_tiles_mat[0,2]=0
    centrecolumn_tiles_mat[1,2]=0
    div_tiles_upper_t1_mat=aff_trans.data['asbuilt/rz_2d/div_tiles_upper_t1/data'].matrix
    div_tiles_upper_t1_mat[0,2]=0
    div_tiles_upper_t1_mat[1,2]=0
    div_tiles_upper_t2_mat=aff_trans.data['asbuilt/rz_2d/div_tiles_upper_t2/data'].matrix
    div_tiles_upper_t2_mat[0,2]=0
    div_tiles_upper_t2_mat[1,2]=0
    div_tiles_upper_t3_mat=aff_trans.data['asbuilt/rz_2d/div_tiles_upper_t3/data'].matrix
    div_tiles_upper_t3_mat[0,2]=0
    div_tiles_upper_t3_mat[1,2]=0
    div_tiles_upper_t4_mat=aff_trans.data['asbuilt/rz_2d/div_tiles_upper_t4/data'].matrix
    div_tiles_upper_t4_mat[0,2]=0
    div_tiles_upper_t4_mat[1,2]=0
    div_tiles_upper_t5_mat=aff_trans.data['asbuilt/rz_2d/div_tiles_upper_t5/data'].matrix
    div_tiles_upper_t5_mat[0,2]=0
    div_tiles_upper_t5_mat[1,2]=0

    div_tiles_lower_t1_mat=aff_trans.data['asbuilt/rz_2d/div_tiles_lower_t1/data'].matrix
    div_tiles_lower_t1_mat[0,2]=0
    div_tiles_lower_t1_mat[1,2]=0
    div_tiles_lower_t2_mat=aff_trans.data['asbuilt/rz_2d/div_tiles_lower_t2/data'].matrix
    div_tiles_lower_t2_mat[0,2]=0
    div_tiles_lower_t2_mat[1,2]=0
    div_tiles_lower_t3_mat=aff_trans.data['asbuilt/rz_2d/div_tiles_lower_t3/data'].matrix
    div_tiles_lower_t3_mat[0,2]=0
    div_tiles_lower_t3_mat[1,2]=0
    div_tiles_lower_t4_mat=aff_trans.data['asbuilt/rz_2d/div_tiles_lower_t4/data'].matrix
    div_tiles_lower_t4_mat[0,2]=0
    div_tiles_lower_t4_mat[1,2]=0
    div_tiles_lower_t5_mat=aff_trans.data['asbuilt/rz_2d/div_tiles_lower_t5/data'].matrix
    div_tiles_lower_t5_mat[0,2]=0
    div_tiles_lower_t5_mat[1,2]=0

    cassette_upper_mat=aff_trans.data['asbuilt/rz_2d/cassette_upper/data'].matrix
    cassette_upper_mat[0,2]=0
    cassette_upper_mat[1,2]=0
    cassette_lower_mat=aff_trans.data['asbuilt/rz_2d/cassette_lower/data'].matrix
    cassette_lower_mat[0,2]=0
    cassette_lower_mat[1,2]=0

    unshifted_RZ_vec=np.array([R_unshifted,Z_unshifted,1]).reshape(3,1)
    if upper_or_lower=='upper':
        if tile=='T1':
            shift_vec=div_tiles_upper_t1_mat@unshifted_RZ_vec
        if tile=='T2':
            shift_vec=div_tiles_upper_t2_mat@unshifted_RZ_vec
        if tile=='T3':
            shift_vec=div_tiles_upper_t3_mat@unshifted_RZ_vec
        if tile=='T4':
            shift_vec=div_tiles_upper_t4_mat@unshifted_RZ_vec
        if tile=='T5':
            shift_vec=div_tiles_upper_t5_mat@unshifted_RZ_vec
        if tile=='C5' or tile=='C6':
            shift_vec=centrecolumn_tiles_mat@unshifted_RZ_vec
        if tile=='B1' or tile=='B2' or tile=='B3' or tile=='B4' or tile=='N1' or tile=='N2':
            shift_vec=cassette_upper_mat@unshifted_RZ_vec
    else:
        if tile=='T1':
            shift_vec=div_tiles_lower_t1_mat@unshifted_RZ_vec
        if tile=='T2':
            shift_vec=div_tiles_lower_t2_mat@unshifted_RZ_vec
        if tile=='T3':
            shift_vec=div_tiles_lower_t3_mat@unshifted_RZ_vec
        if tile=='T4':
            shift_vec=div_tiles_lower_t4_mat@unshifted_RZ_vec
        if tile=='T5':
            shift_vec=div_tiles_lower_t5_mat@unshifted_RZ_vec
        if tile=='C5' or tile=='C6':
            shift_vec=centrecolumn_tiles_mat@unshifted_RZ_vec
        if tile=='B1' or tile=='B2' or tile=='B3' or tile=='B4' or tile=='N1' or tile=='N2':
            shift_vec=cassette_lower_mat@unshifted_RZ_vec
    R_shifted=shift_vec[0][0]
    Z_shifted=shift_vec[1][0]
    return R_shifted, Z_shifted


def affine_cylindrical_vector(unshifted_rTz_lower,unshifted_rTz_upper,tile):
    probe_surface_normal_r_shifted_upper, probe_surface_normal_z_shifted_upper=RZ_shifts(unshifted_rTz_upper[0],unshifted_rTz_upper[2],tile,'upper')
    mag_rz_unshifted_upper=(unshifted_rTz_upper[0]**2+unshifted_rTz_upper[2]**2)**0.5 #unnecessary I think..
    scaling_RZ_upper=mag_rz_unshifted_upper/(probe_surface_normal_r_shifted_upper**2+probe_surface_normal_z_shifted_upper**2)**0.5
    shifted_rTz_upper=np.array([scaling_RZ_upper*probe_surface_normal_r_shifted_upper, unshifted_rTz_upper[1] ,scaling_RZ_upper*probe_surface_normal_z_shifted_upper ])

    probe_surface_normal_r_shifted_lower, probe_surface_normal_z_shifted_lower=RZ_shifts(unshifted_rTz_lower[0],unshifted_rTz_lower[2],tile,'lower')
    mag_rz_unshifted_lower=(unshifted_rTz_lower[0]**2+unshifted_rTz_lower[2]**2)**0.5 #unnecessary I think..
    scaling_RZ_lower=mag_rz_unshifted_lower/(probe_surface_normal_r_shifted_lower**2+probe_surface_normal_z_shifted_lower**2)**0.5
    shifted_rTz_lower=np.array([scaling_RZ_lower*probe_surface_normal_r_shifted_lower, unshifted_rTz_lower[1] ,scaling_RZ_lower*probe_surface_normal_z_shifted_lower ])
    return shifted_rTz_lower,shifted_rTz_upper


##PERFORM AFFINE TRANSFORMATIONS

#assume toroidal component points anticlockwise when looking from upper to lower. Deduced this from csv files of tiles.
#########################################################################################
#T2 probes
#45 degree angle in RZ plane, 10 degrees in toroidal direction.

#rz plane vector for lower
rz_SN=np.array([1,0,1])/(2**0.5)
probe_angle_wrt_rz_normal=10*np.pi/180
scaling=np.cos(probe_angle_wrt_rz_normal)
probe_normal_theta=(1-(scaling**2)*(rz_SN[0]**2+rz_SN[2]**2))**0.5
T2_probe_surface_normal_rTz_lower=np.array([   scaling*rz_SN[0] , probe_normal_theta , scaling*rz_SN[2]  ])
T2_probe_surface_normal_rTz_upper=np.array([   scaling*rz_SN[0] , probe_normal_theta , -1*scaling*rz_SN[2]  ])

T2_probe_surface_normal_rTz_lower_shifted,T2_probe_surface_normal_rTz_upper_shifted=affine_cylindrical_vector(T2_probe_surface_normal_rTz_lower,T2_probe_surface_normal_rTz_upper,'T2')


#####################################################################################################
#T3 probes
#45-0.444 degree angle in RZ plane, 10 degrees in toroidal direction.
angle_RZ=(45-0.444)*np.pi/180
radial_component_par_tile=np.cos(angle_RZ)
z_component_par_tile=(1-radial_component_par_tile**2)**0.5

z_component_tile_normal=radial_component_par_tile*(z_component_par_tile**2+radial_component_par_tile**2)**-0.5
radial_component_tile_normal=(1-z_component_tile_normal**2)**0.5

rz_SN=np.array([radial_component_tile_normal,0,z_component_tile_normal])
probe_angle_wrt_rz_normal=10*np.pi/180
scaling=np.cos(probe_angle_wrt_rz_normal)
probe_normal_theta=(1-(scaling**2)*(rz_SN[0]**2+rz_SN[2]**2))**0.5
T3_probe_surface_normal_rTz_lower=np.array([   scaling*rz_SN[0] , probe_normal_theta , scaling*rz_SN[2]  ])
T3_probe_surface_normal_rTz_upper=np.array([   scaling*rz_SN[0] , probe_normal_theta , -1*scaling*rz_SN[2]  ])

T3_probe_surface_normal_rTz_lower_shifted,T3_probe_surface_normal_rTz_upper_shifted=affine_cylindrical_vector(T3_probe_surface_normal_rTz_lower,T3_probe_surface_normal_rTz_upper,'T3')

################################################################
#T4 probes
#45 degree angle in RZ plane, 10 degrees in toroidal direction.

#rz plane vector for lower
rz_SN=np.array([0,0,1])
probe_angle_wrt_rz_normal=10*np.pi/180
scaling=np.cos(probe_angle_wrt_rz_normal)
probe_normal_theta=(1-(scaling**2)*(rz_SN[0]**2+rz_SN[2]**2))**0.5
T4_probe_surface_normal_rTz_lower=np.array([   scaling*rz_SN[0] , probe_normal_theta , scaling*rz_SN[2]  ])
T4_probe_surface_normal_rTz_upper=np.array([   scaling*rz_SN[0] , probe_normal_theta , -1*scaling*rz_SN[2]  ])
T4_probe_surface_normal_rTz_lower_shifted,T4_probe_surface_normal_rTz_upper_shifted=affine_cylindrical_vector(T4_probe_surface_normal_rTz_lower,T4_probe_surface_normal_rTz_upper,'T4')

################################################################
#T5 probes
#45 degree angle in RZ plane, 10 degrees in toroidal direction, negative radial component

#rz plane vector for lower
rz_SN=np.array([-1,0,1])/(2**0.5)
probe_angle_wrt_rz_normal=10*np.pi/180
scaling=np.cos(probe_angle_wrt_rz_normal)
probe_normal_theta=(1-(scaling**2)*(rz_SN[0]**2+rz_SN[2]**2))**0.5
T5_probe_surface_normal_rTz_lower=np.array([   scaling*rz_SN[0] , probe_normal_theta , scaling*rz_SN[2]  ])
T5_probe_surface_normal_rTz_upper=np.array([   scaling*rz_SN[0] , probe_normal_theta , -1*scaling*rz_SN[2]  ])
T5_probe_surface_normal_rTz_lower_shifted,T5_probe_surface_normal_rTz_upper_shifted=affine_cylindrical_vector(T5_probe_surface_normal_rTz_lower,T5_probe_surface_normal_rTz_upper,'T5')

#######################################################
#N1 probes

#parallel to tile in RZ plane. Taken from master geometry document
point1_z=-1007
point1_r=1190.687

point2_z=-1377.687
point2_r=820

r_diff=(point1_r-point2_r)
z_diff=(point1_z-point2_z)

radial_component_par_tile=r_diff/(r_diff**2+z_diff**2)**0.5
z_component_par_tile=z_diff/(r_diff**2+z_diff**2)**0.5

z_component_tile_normal=radial_component_par_tile*(z_component_par_tile**2+radial_component_par_tile**2)**-0.5
radial_component_tile_normal=-1*(1-z_component_tile_normal**2)**0.5

rz_SN=np.array([radial_component_tile_normal,0,z_component_tile_normal])
probe_angle_wrt_rz_normal=3.5*np.pi/180
scaling=np.cos(probe_angle_wrt_rz_normal)
probe_normal_theta=(1-(scaling**2)*(rz_SN[0]**2+rz_SN[2]**2))**0.5
N1_probe_surface_normal_rTz_lower=np.array([   scaling*rz_SN[0] , probe_normal_theta , scaling*rz_SN[2]  ])
N1_probe_surface_normal_rTz_upper=np.array([   scaling*rz_SN[0] , probe_normal_theta , -1*scaling*rz_SN[2]  ])
N1_probe_surface_normal_rTz_lower_shifted,N1_probe_surface_normal_rTz_upper_shifted=affine_cylindrical_vector(N1_probe_surface_normal_rTz_lower,N1_probe_surface_normal_rTz_upper,'N1')

###########################################################################################
#N2
angle_wrt_z=90-10.9
r_diff=3*np.cos(angle_wrt_z*np.pi/180)
z_diff=3*np.sin(angle_wrt_z*np.pi/180)

radial_component_par_tile=r_diff/(r_diff**2+z_diff**2)**0.5
z_component_par_tile=z_diff/(r_diff**2+z_diff**2)**0.5

z_component_tile_normal=radial_component_par_tile*(z_component_par_tile**2+radial_component_par_tile**2)**-0.5
radial_component_tile_normal=-1*(1-z_component_tile_normal**2)**0.5

rz_SN=np.array([radial_component_tile_normal,0,z_component_tile_normal])
probe_angle_wrt_rz_normal=10*np.pi/180
scaling=np.cos(probe_angle_wrt_rz_normal)
probe_normal_theta=(1-(scaling**2)*(rz_SN[0]**2+rz_SN[2]**2))**0.5
N2_probe_surface_normal_rTz_lower=np.array([   scaling*rz_SN[0] , probe_normal_theta , scaling*rz_SN[2]  ])
N2_probe_surface_normal_rTz_upper=np.array([   scaling*rz_SN[0] , probe_normal_theta , -1*scaling*rz_SN[2]  ])
N2_probe_surface_normal_rTz_lower_shifted,N2_probe_surface_normal_rTz_upper_shifted=affine_cylindrical_vector(N2_probe_surface_normal_rTz_lower,N2_probe_surface_normal_rTz_upper,'N2')

###############################################################################################
#Baffles
#parallel to tile in RZ plane. Taken from master geometry document
point1_z=-1599
point1_r=820

point2_z=-1559.487
point2_r=1725

r_diff=(point1_r-point2_r)
z_diff=(point1_z-point2_z)

radial_component_par_tile=r_diff/(r_diff**2+z_diff**2)**0.5
z_component_par_tile=z_diff/(r_diff**2+z_diff**2)**0.5

z_component_tile_normal=radial_component_par_tile*(z_component_par_tile**2+radial_component_par_tile**2)**-0.5
radial_component_tile_normal=(1-z_component_tile_normal**2)**0.5

B_probe_surface_normal_rTz_lower=np.array([  radial_component_tile_normal , 0 , z_component_tile_normal ])
B_probe_surface_normal_rTz_upper=np.array([  radial_component_tile_normal , 0 , -1*z_component_tile_normal ])
#probes are flush with the surface.

B_probe_surface_normal_rTz_lower_shifted,B_probe_surface_normal_rTz_upper_shifted=affine_cylindrical_vector(B_probe_surface_normal_rTz_lower,B_probe_surface_normal_rTz_upper,'B4')
#same shifts for all B tiles.
#############################################################################################
#C6
rz_SN=np.array([1,0,0])
probe_angle_wrt_rz_normal=7.3*np.pi/180
scaling=np.cos(probe_angle_wrt_rz_normal)
probe_normal_theta=(1-(scaling**2)*(rz_SN[0]**2+rz_SN[2]**2))**0.5
C6_probe_surface_normal_rTz_lower=np.array([   scaling*rz_SN[0] , probe_normal_theta , scaling*rz_SN[2]  ])
C6_probe_surface_normal_rTz_upper=np.array([   scaling*rz_SN[0] , probe_normal_theta , -1*scaling*rz_SN[2]  ])
C6_probe_surface_normal_rTz_lower_shifted,C6_probe_surface_normal_rTz_upper_shifted=affine_cylindrical_vector(C6_probe_surface_normal_rTz_lower,C6_probe_surface_normal_rTz_upper,'C6')

#############################################################################################################
#C5
point1_z=-853.5
point1_r=303.3

point2_z=-1100
point2_r= 335

r_diff=(point1_r-point2_r)
z_diff=(point1_z-point2_z)

radial_component_par_tile=-1*r_diff/(r_diff**2+z_diff**2)**0.5
z_component_par_tile=z_diff/(r_diff**2+z_diff**2)**0.5

z_component_tile_normal=radial_component_par_tile*(z_component_par_tile**2+radial_component_par_tile**2)**-0.5
radial_component_tile_normal=(1-z_component_tile_normal**2)**0.5
rz_SN=np.array([radial_component_tile_normal,0,z_component_tile_normal])

probe_angle_wrt_rz_normal=7.3*np.pi/180
scaling=np.cos(probe_angle_wrt_rz_normal)
probe_normal_theta=(1-(scaling**2)*(rz_SN[0]**2+rz_SN[2]**2))**0.5
C5_probe_surface_normal_rTz_lower=np.array([   scaling*rz_SN[0] , probe_normal_theta , scaling*rz_SN[2]  ])
C5_probe_surface_normal_rTz_upper=np.array([   scaling*rz_SN[0] , probe_normal_theta , -1*scaling*rz_SN[2]  ])
C5_probe_surface_normal_rTz_lower_shifted,C5_probe_surface_normal_rTz_upper_shifted=affine_cylindrical_vector(C5_probe_surface_normal_rTz_lower,C5_probe_surface_normal_rTz_upper,'C5')
