import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import pyuda
client=pyuda.Client()

#example

d_theta=1e-3 #probe-tile separation in m
d_perp=0.2e-3 #height of bottom of probe relative to top of tile in m
theta_p=10*np.pi/180

probe_tile_SN_rTz=np.array([0.70706184, 0.0109084 , 0.70706757])
B_rTz_rough=np.array([0.70706184, -50 , 0.70706757])
B_rTz=B_rTz_rough/((B_rTz_rough[0]**2+B_rTz_rough[1]**2+B_rTz_rough[2]**2)**0.5)
probe_tile_SN_rz=np.array([probe_tile_SN_rTz[0],0,probe_tile_SN_rTz[2]])/((probe_tile_SN_rTz[0]**2+probe_tile_SN_rTz[2]**2)**0.5)

angle_rz_and_B=np.abs(np.arccos(np.dot(B_rTz,probe_tile_SN_rz)))
theta_perp=(np.pi/2)-angle_rz_and_B
print(theta_perp*180/np.pi)

d=(d_perp-d_theta*np.tan(theta_perp))/(np.cos(theta_p)*np.tan(theta_perp)+np.sin(theta_p))
print(d)
