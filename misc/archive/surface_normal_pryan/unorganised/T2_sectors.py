import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

def T2_surface_normal(position_rthetaz):
    position_x=position_rthetaz[0]*np.cos(position_rthetaz[1]) #convert back to xyz coordinates.
    position_y=position_rthetaz[0]*np.sin(position_rthetaz[1])
    position_z=position_rthetaz[2]
    position_xyz=np.array([position_x,position_y,position_z])

########################################################################################
############################################################################################
#there are 24 T2 plates in the toroidal direction.
#the angular coverage of each T2 tile is 2*np.pi/24
#on each tile, there is a constant surface normal.
#firstly we need to calculate which of the 24 tiles position_xyz (or equivalently position_rthetaz) lies on.
#THIS SECTION CAN BE IMPROVED - does not consider the exact location of the T2 surface in the machine.

#theta increases anticlockwise from 3 o'clock looking from upper to lower divertor in plan view
    step=2*np.pi/24
    upper_intervals=np.transpose(np.array([np.linspace(step,2*np.pi,24)]))
    lower_intervals=np.transpose(np.array([np.linspace(0,(2*np.pi)-step,24)]))
    intervals=np.concatenate([lower_intervals,upper_intervals],axis=1) #each interval is for a different T2 tile. There are 24 surface-normal vectors for T2 in xyz basis.
#note that T2 tiles are not arranged in these intervals.. There is an offset!

    relative_theta=position_rthetaz[1]-(-step)  #the tile from MU00859 does not start cover theta range: 0 to step radians. The centre of the reference T2 tile is approximately at theta=-step/2
#Script needs angle relative to this reference tile.
    theta_shifts=np.linspace(0,2*np.pi,24)   #this specifies the rotation of each surface normal about the machine axis; corresponds to relative_intervals

     #rotation required at the position we want to calculate surface normal at.
    for i in range(24):
        if (relative_theta>intervals[i][0]) & (relative_theta<=intervals[i][1]):
            rotation_theta=theta_shifts[i]
            break
    if rotation_theta==0:
        rotation_theta=2*np.pi

#we need to rotate reference surface normal by angle rotation_theta about the machine z axis (centre column)

######################################################################################
#######################################################################################

#Reference surface normal from MU00859
    #normal_reference=np.array([0.013671896,-0.001887431,0.013815705]) #x,y,z basis
    normal_reference=np.array([0.70797412,-0.0754802,0.702193268]) #x,y,z basis


#calculate surface normal at position_rthetaz. z component is independent of toroidal angle.
    surface_normal_x=normal_reference[0]*np.cos(rotation_theta)-normal_reference[1]*np.sin(rotation_theta) #used trig identity cos(a+b)=..and sin(a+b)=..
    surface_normal_y=normal_reference[0]*np.sin(rotation_theta)+normal_reference[1]*np.cos(rotation_theta)
    surface_normal_xyz=np.array([surface_normal_x,surface_normal_y,normal_reference[2]])
##########################################
#######################################
#change basis to radial, toroidal, axial (all in units of m)
    radial_unit_vector=np.array([position_xyz[0],position_xyz[1],0])/((position_xyz[0]**2+position_xyz[1]**2)**0.5)

    toroidal_unit_vector_magnitude_y=np.abs(radial_unit_vector[0]/(radial_unit_vector[0]**2+radial_unit_vector[1]**2)**0.5)
    toroidal_unit_vector_magnitude_x=np.abs((1-toroidal_unit_vector_magnitude_y**2)**0.5)

    if (rotation_theta>0) & (rotation_theta<=np.pi/2):
        toroidal_unit_vector=np.array([-toroidal_unit_vector_magnitude_x,toroidal_unit_vector_magnitude_y,0])
    if (rotation_theta>np.pi/2) & (rotation_theta<=np.pi):
        toroidal_unit_vector=np.array([-toroidal_unit_vector_magnitude_x,-toroidal_unit_vector_magnitude_y,0])
    if (rotation_theta>np.pi) & (rotation_theta<=3*np.pi/2):
        toroidal_unit_vector=np.array([toroidal_unit_vector_magnitude_x,-toroidal_unit_vector_magnitude_y,0])
    if (rotation_theta>3*np.pi/2) & (rotation_theta<=2*np.pi):
        toroidal_unit_vector=np.array([toroidal_unit_vector_magnitude_x,toroidal_unit_vector_magnitude_y,0])

#############################################################
    if np.abs(radial_unit_vector[0])>1e-10: #problem with infinity when small x component of radial unit vector
        surface_normal_toroidal=(surface_normal_xyz[1]-(radial_unit_vector[1]*surface_normal_xyz[0]/radial_unit_vector[0]))/(toroidal_unit_vector[1]-(toroidal_unit_vector[0]*radial_unit_vector[1]/radial_unit_vector[0]))
        surface_normal_radial=(surface_normal_xyz[0]-(surface_normal_toroidal*toroidal_unit_vector[0]))/radial_unit_vector[0]
    elif np.abs(radial_unit_vector[0])<1e-10:
        surface_normal_toroidal=surface_normal_xyz[0]/toroidal_unit_vector[0]
        surface_normal_radial=(surface_normal_xyz[1]-(toroidal_unit_vector[1]*surface_normal_toroidal)/radial_unit_vector[1])

    surface_normal_rTz=np.array([surface_normal_radial,surface_normal_toroidal,normal_reference[2]]) #equivalent to surface_normal_xyz but in new basis.
    magnitude_rTz=(surface_normal_rTz[0]**2+surface_normal_rTz[1]**2+surface_normal_rTz[2]**2)**0.5
    print(magnitude_rTz) #check equal to 1.
#I think B field will be returned in terms of Br, BT, Bz so it is good to have the surface normals in this basis!
    z_unit_vector=np.array([0,0,1])
    return surface_normal_xyz,surface_normal_rTz, radial_unit_vector, toroidal_unit_vector, z_unit_vector, position_xyz

#########################################################################################################
#########################################################################################################
#test
position_xyz=np.array([0,1,-1]) #calculate surface normal at this position
radial_position=(position_xyz[0]**2+position_xyz[1]**2)**0.5
theta=np.arctan(position_xyz[1]/position_xyz[0])
position_rthetaz=np.array([radial_position,theta,position_xyz[2]])
norm_xyz, norm_rTz, r_unit, T_unit, z_unit, position_xyz = T2_surface_normal(position_rthetaz)

#check that basis conversion is correct
check_xyz=norm_rTz[0]*r_unit+norm_rTz[1]*T_unit+norm_rTz[2]*z_unit
print(check_xyz-norm_xyz) #should equal zero vector


#####################################
#lets do some plotting..
filename='T2.csv'
T2_surface=pd.read_csv(filename)
x_surface=T2_surface['x']
y_surface=T2_surface['y']
z_surface=T2_surface['z']


step=2*np.pi/24
position_reference=np.array([0.7,-step/2,-1.97]) #a position on the tile from MU00859 #r, angle, z
fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.plot(x_surface,y_surface,z_surface,'o',linestyle='')
for i in range(24):
    position=np.array([position_reference[0],position_reference[1]+(i*step),position_reference[2]]) #r, angle, z
    norm_xyz, norm_rTz, r_unit, T_unit, z_unit, position_xyz = T2_surface_normal(position)
    x_points=np.array([0,1*norm_xyz[0]])+position_xyz[0]
    y_points=np.array([0,1*norm_xyz[1]])+position_xyz[1]
    z_points=np.array([0,1*norm_xyz[2]])+position_xyz[2]
    print(norm_rTz)
    plt.plot(x_points,y_points,z_points)
plt.show()
#this should plot surface normal at same location on a T2 tile, for all 24 tiles.
