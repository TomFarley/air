import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import pyuda
client=pyuda.Client()



#assume toroidal component points anticlockwise when looking from upper to lower. Deduced this from csv files of tiles.


#########################################################################################
#T2 probes
#45 degree angle in RZ plane, 10 degrees in toroidal direction.

#rz plane vector for lower
rz_SN=np.array([1,0,1])/(2**0.5)
probe_angle_wrt_rz_normal=10*np.pi/180
scaling=np.cos(probe_angle_wrt_rz_normal)
probe_normal_theta=(1-(scaling**2)*(rz_SN[0]**2+rz_SN[2]**2))**0.5
T2_probe_surface_normal_rTz=np.array([   scaling*rz_SN[0] , probe_normal_theta , scaling*rz_SN[2]  ])

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
T3_probe_surface_normal_rTz=np.array([   scaling*rz_SN[0] , probe_normal_theta , scaling*rz_SN[2]  ])



################################################################
#T4 probes
#45 degree angle in RZ plane, 10 degrees in toroidal direction.

#rz plane vector for lower
rz_SN=np.array([0,0,1])
probe_angle_wrt_rz_normal=10*np.pi/180
scaling=np.cos(probe_angle_wrt_rz_normal)
probe_normal_theta=(1-(scaling**2)*(rz_SN[0]**2+rz_SN[2]**2))**0.5
T4_probe_surface_normal_rTz=np.array([   scaling*rz_SN[0] , probe_normal_theta , scaling*rz_SN[2]  ])


################################################################
#T5 probes
#45 degree angle in RZ plane, 10 degrees in toroidal direction, negative radial component

#rz plane vector for lower
rz_SN=np.array([-1,0,1])/(2**0.5)
probe_angle_wrt_rz_normal=10*np.pi/180
scaling=np.cos(probe_angle_wrt_rz_normal)
probe_normal_theta=(1-(scaling**2)*(rz_SN[0]**2+rz_SN[2]**2))**0.5
T5_probe_surface_normal_rTz=np.array([   scaling*rz_SN[0] , probe_normal_theta , scaling*rz_SN[2]  ])


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
N1_probe_surface_normal_rTz=np.array([   scaling*rz_SN[0] , probe_normal_theta , scaling*rz_SN[2]  ])
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
N2_probe_surface_normal_rTz=np.array([   scaling*rz_SN[0] , probe_normal_theta , scaling*rz_SN[2]  ])

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

B_probe_surface_normal_rTz=np.array([  radial_component_tile_normal , 0 , z_component_tile_normal ])
#probes are flush with the surface.
#############################################################################################
#C6
rz_SN=np.array([1,0,0])
probe_angle_wrt_rz_normal=7.3*np.pi/180
scaling=np.cos(probe_angle_wrt_rz_normal)
probe_normal_theta=(1-(scaling**2)*(rz_SN[0]**2+rz_SN[2]**2))**0.5
C6_probe_surface_normal_rTz=np.array([   scaling*rz_SN[0] , probe_normal_theta , scaling*rz_SN[2]  ])

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
C5_probe_surface_normal_rTz=np.array([   scaling*rz_SN[0] , probe_normal_theta , scaling*rz_SN[2]  ])
