import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import pyuda
client=pyuda.Client()


def tile_rows(filename):
    surface=pd.read_csv(filename)
    x_surface=np.array(surface['x'],dtype = np.float32)
    y_surface=np.array(surface['y'],dtype = np.float32)
    z_surface=np.array(surface['z'],dtype = np.float32)

    x_diff=np.diff(x_surface)
    y_diff=np.diff(y_surface)
    z_diff=np.diff(z_surface)

    distances=(x_diff**2+y_diff**2+z_diff**2)**0.5 #distance between adjacent points
    new_row_index,=np.where(distances>0.05) #new row index
    x_data_rows=[]
    x_data_rows.append(x_surface[0:new_row_index[0]])
    y_data_rows=[]
    y_data_rows.append(y_surface[0:new_row_index[0]])
    z_data_rows=[]
    z_data_rows.append(z_surface[0:new_row_index[0]])
    for i in range(len(new_row_index)-1):
        x_row=x_surface[new_row_index[i]+1:new_row_index[i+1]]
        x_data_rows.append(x_row)
        y_row=y_surface[new_row_index[i]+1:new_row_index[i+1]]
        y_data_rows.append(y_row)
        z_row=z_surface[new_row_index[i]+1:new_row_index[i+1]]
        z_data_rows.append(z_row)
    return x_data_rows, y_data_rows, z_data_rows

def three_points(x_data_rows, y_data_rows, z_data_rows, row_index, column_index):
    interest_x=x_data_rows[row_index][column_index]
    interest_y=y_data_rows[row_index][column_index]
    interest_z=z_data_rows[row_index][column_index]
    point1=np.array([interest_x,interest_y,interest_z])

    if row_index==0:
        two_row_x=x_data_rows[1]
        two_row_y=y_data_rows[1]
        two_row_z=z_data_rows[1]
        diff_x=interest_x-two_row_x
        diff_y=interest_y-two_row_y
        diff_z=interest_z-two_row_z
        distances=(diff_x**2+diff_y**2+diff_z**2)**0.5
        indices_sort=np.argsort(distances)
    if (row_index==len(x_data_rows)-1) or (row_index==-1):
        two_row_x=x_data_rows[-2]
        two_row_y=y_data_rows[-2]
        two_row_z=z_data_rows[-2]
        diff_x=interest_x-two_row_x
        diff_y=interest_y-two_row_y
        diff_z=interest_z-two_row_z
        distances=(diff_x**2+diff_y**2+diff_z**2)**0.5
        indices_sort=np.argsort(distances)
    if (row_index>0) and (row_index<len(x_data_rows)-1):
        above_x=x_data_rows[row_index+1]
        above_y=y_data_rows[row_index+1]
        above_z=z_data_rows[row_index+1]

        abv_diff_x=interest_x-above_x
        abv_diff_y=interest_y-above_y
        abv_diff_z=interest_z-above_z
        abv_distances=(abv_diff_x**2+abv_diff_y**2+abv_diff_z**2)**0.5
        min_abv=min(abv_distances)

        below_x=x_data_rows[row_index-1]
        below_y=y_data_rows[row_index-1]
        below_z=z_data_rows[row_index-1]

        blw_diff_x=interest_x-below_x
        blw_diff_y=interest_y-below_y
        blw_diff_z=interest_z-below_z
        blw_distances=(blw_diff_x**2+blw_diff_y**2+blw_diff_z**2)**0.5
        min_blw=min(blw_distances)

        if min_abv>min_blw:
            indices_sort=np.argsort(abv_distances)
            two_row_x=x_data_rows[row_index+1]
            two_row_y=y_data_rows[row_index+1]
            two_row_z=z_data_rows[row_index+1]
        else:
            indices_sort=np.argsort(abv_distances)
            two_row_x=x_data_rows[row_index-1]
            two_row_y=y_data_rows[row_index-1]
            two_row_z=z_data_rows[row_index-1]

    point2=np.array([two_row_x[indices_sort[0]],two_row_y[indices_sort[0]],two_row_z[indices_sort[0]]])
    point3=np.array([two_row_x[indices_sort[1]],two_row_y[indices_sort[1]],two_row_z[indices_sort[1]]])

    return point1, point2, point3

def surface_normal_xyz_fn(point1,point2,point3):
    surface_vector1=point1-point2
    surface_vector2=point1-point3
    one_cross_two=np.cross(surface_vector1,surface_vector2)
    magnitude=(one_cross_two[0]**2+one_cross_two[1]**2+one_cross_two[2]**2)**0.5
    surface_normal_xyz=-1*one_cross_two/magnitude
    return surface_normal_xyz

def calculate_cylindrical_unit_vectors(point1):
    radial_unit_vector=np.array([point1[0],point1[1],0])/((point1[0]**2+point1[1]**2)**0.5)

    toroidal_unit_vector_magnitude_y=np.abs(radial_unit_vector[0]/(radial_unit_vector[0]**2+radial_unit_vector[1]**2)**0.5)
    toroidal_unit_vector_magnitude_x=np.abs((1-toroidal_unit_vector_magnitude_y**2)**0.5)

    if (point1[0]>0) & (point1[1]>0):
        toroidal_unit_vector=np.array([-toroidal_unit_vector_magnitude_x,toroidal_unit_vector_magnitude_y,0])
    if (point1[0]<0) & (point1[1]>0):
        toroidal_unit_vector=np.array([-toroidal_unit_vector_magnitude_x,-toroidal_unit_vector_magnitude_y,0])
    if (point1[0]<0) & (point1[1]<0):
        toroidal_unit_vector=np.array([toroidal_unit_vector_magnitude_x,-toroidal_unit_vector_magnitude_y,0])
    if (point1[0]>0) & (point1[1]<0):
        toroidal_unit_vector=np.array([toroidal_unit_vector_magnitude_x,toroidal_unit_vector_magnitude_y,0])

    z_unit_vector=np.array([0,0,1])
    cylindrical_unit_vectors=np.array([radial_unit_vector, toroidal_unit_vector, z_unit_vector])
    return cylindrical_unit_vectors


def cylindrical_transform_surface_normal_xyz(surface_normal_xyz,cylindrical_unit_vectors,tile):
    radial_unit_vector=cylindrical_unit_vectors[0]
    toroidal_unit_vector=cylindrical_unit_vectors[1]
    if np.abs(radial_unit_vector[0])>1e-10: #problem with infinity when small x component of radial unit vector
        surface_normal_toroidal=(surface_normal_xyz[1]-(radial_unit_vector[1]*surface_normal_xyz[0]/radial_unit_vector[0]))/(toroidal_unit_vector[1]-(toroidal_unit_vector[0]*radial_unit_vector[1]/radial_unit_vector[0]))
        surface_normal_radial=(surface_normal_xyz[0]-(surface_normal_toroidal*toroidal_unit_vector[0]))/radial_unit_vector[0]
    elif np.abs(radial_unit_vector[0])<1e-10:
        surface_normal_toroidal=surface_normal_xyz[0]/toroidal_unit_vector[0]
        surface_normal_radial=(surface_normal_xyz[1]-(toroidal_unit_vector[1]*surface_normal_toroidal)/radial_unit_vector[1])
    surface_normal_rTz=np.array([surface_normal_radial,surface_normal_toroidal,surface_normal_xyz[2]]) #equivalent to surface_normal_xyz but in new basis.
    magnitude_rTz=(surface_normal_rTz[0]**2+surface_normal_rTz[1]**2+surface_normal_rTz[2]**2)**0.5
    print(magnitude_rTz)
    if (tile=='T5') & (surface_normal_radial>0):
        surface_normal_rTz=surface_normal_rTz*-1
    if (tile=='T2') & (surface_normal_radial<0):
        surface_normal_rTz=surface_normal_rTz*-1
    if (tile=='T3') & (surface_normal_radial<0):
        surface_normal_rTz=surface_normal_rTz*-1
    if (tile=='T4') & (surface_normal_xyz[2]<0):
        surface_normal_rTz=surface_normal_rTz*-1
    return surface_normal_rTz


def create_vector_data_cyl(surface_normal_rTz, cylindrical_unit_vectors, point1):
    radial_unit_vector=cylindrical_unit_vectors[0]
    toroidal_unit_vector=cylindrical_unit_vectors[1]
    z_unit_vector=cylindrical_unit_vectors[2]
    surface_normal_xyz=surface_normal_rTz[0]*radial_unit_vector+surface_normal_rTz[1]*toroidal_unit_vector+surface_normal_rTz[2]*z_unit_vector
    x_points=np.array([0,1*surface_normal_xyz[0]])+point1[0]
    y_points=np.array([0,1*surface_normal_xyz[1]])+point1[1]
    z_points=np.array([0,1*surface_normal_xyz[2]])+point1[2]
    vector_data=[x_points, y_points, z_points]
    return vector_data

def toroidal_tilt_wrt_xy(surface_normal_rTz, cylindrical_unit_vectors):
    y_unit=np.array([0,1,0]) #cylindrical basis
    surface_normal_Tz=np.array([0,surface_normal_rTz[1],surface_normal_rTz[2]]) #vector in Tz plane
    surface_normal_Tz_cart=cylindrical_unit_vectors[0]*surface_normal_Tz[0]+cylindrical_unit_vectors[1]*surface_normal_Tz[1]+cylindrical_unit_vectors[2]*surface_normal_Tz[2]
    mag_surface_normal_Tz_cart=(surface_normal_Tz_cart[0]**2+surface_normal_Tz_cart[1]**2+surface_normal_Tz_cart[2]**2)**0.5
    dot_product=np.dot(y_unit,surface_normal_Tz_cart)
    tilt_angle=180*np.arccos(dot_product/mag_surface_normal_Tz_cart)/np.pi #degrees
    return tilt_angle #90 dgrees is normal incidence


def cartesian_to_cylindrical_position_coordinates(filename):
    surface=pd.read_csv(filename)
    x_surface=np.array(surface['x'],dtype = np.float32)
    y_surface=np.array(surface['y'],dtype = np.float32)
    z_surface=np.array(surface['z'],dtype = np.float32)
    radius_surface=(x_surface**2+y_surface**2)**0.5
    theta_surface=np.arcsin(x_surface/radius_surface)
    return radius_surface, theta_surface, z_surface


def tile_unit_components_rz(surface_normal_rTz):
    rz_normal=np.array([surface_normal_rTz[0],0,surface_normal_rTz[2]])/(surface_normal_rTz[0]**2+surface_normal_rTz[2]**2)**0.5
    return rz_normal

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

    R_shifted=np.zeros((len(R_unshifted),1))
    Z_shifted=np.zeros((len(Z_unshifted),1))
    for i in range(len(R_unshifted)):
        unshifted_RZ_vec=np.array([R_unshifted[i],Z_unshifted[i],1]).reshape(3,1)
        if upper_or_lower[i]=='upper':
            if tile[i]=='T1':
                shift_vec=div_tiles_upper_t1_mat@unshifted_RZ_vec
            if tile[i]=='T2':
                shift_vec=div_tiles_upper_t2_mat@unshifted_RZ_vec
            if tile[i]=='T3':
                shift_vec=div_tiles_upper_t3_mat@unshifted_RZ_vec
            if tile[i]=='T4':
                shift_vec=div_tiles_upper_t4_mat@unshifted_RZ_vec
            if tile[i]=='T5':
                shift_vec=div_tiles_upper_t5_mat@unshifted_RZ_vec
            if tile[i]=='C5' or tile[i]=='C6':
                shift_vec=centrecolumn_tiles_mat@unshifted_RZ_vec
            if tile[i]=='B1' or tile[i]=='B2' or tile[i]=='B3' or tile[i]=='B4' or tile[i]=='N1' or tile[i]=='N2':
                shift_vec=cassette_upper_mat@unshifted_RZ_vec
        else:
            if tile[i]=='T1':
                shift_vec=div_tiles_lower_t1_mat@unshifted_RZ_vec
            if tile[i]=='T2':
                shift_vec=div_tiles_lower_t2_mat@unshifted_RZ_vec
            if tile[i]=='T3':
                shift_vec=div_tiles_lower_t3_mat@unshifted_RZ_vec
            if tile[i]=='T4':
                shift_vec=div_tiles_lower_t4_mat@unshifted_RZ_vec
            if tile[i]=='T5':
                shift_vec=div_tiles_lower_t5_mat@unshifted_RZ_vec
            if tile[i]=='C5' or tile[i]=='C6':
                shift_vec=centrecolumn_tiles_mat@unshifted_RZ_vec
            if tile[i]=='B1' or tile[i]=='B2' or tile[i]=='B3' or tile[i]=='B4' or tile[i]=='N1' or tile[i]=='N2':
                shift_vec=cassette_lower_mat@unshifted_RZ_vec
        R_shifted[i]=shift_vec[0][0]
        Z_shifted[i]=shift_vec[1][0]
    return R_shifted, Z_shifted



########################################
#####################################

rotation_angle=45*np.pi/180
shiftr=0
shiftz=0
test_matrix=np.array([(np.cos(rotation_angle),np.sin(rotation_angle),shiftr),(-np.sin(rotation_angle),np.cos(rotation_angle),shiftz),(0,0,1)])

radius=np.concatenate([(np.linspace(0,30,51)),(np.ones((51))*0)])
axial=np.concatenate([(np.ones((51))*0),(np.linspace(-40,60,51))])
dummy=np.ones((102))
shift_radius=np.array([])
shift_axial=np.array([])
for i in range(102):
    unshift_vector=np.array([(radius[i]),(axial[i]),(dummy[i])]).reshape(3,1)
    shift_vector=test_matrix@unshift_vector
    shift_radius=np.concatenate((shift_radius,shift_vector[0]))
    shift_axial=np.concatenate((shift_axial,shift_vector[1]))
plt.figure()
plt.plot(radius,axial,'o',color='r', linestyle='')
plt.plot(shift_radius,shift_axial,'o', linestyle='')
plt.xlim(-70,70)
plt.ylim(-70,70)
plt.show()

###############################################
################################################
tile='T2'
filename=tile+'.csv'

x_data_rows, y_data_rows, z_data_rows=tile_rows(filename)
point1, point2, point3=three_points(x_data_rows, y_data_rows, z_data_rows, 30,0)
surface_normal_xyz=surface_normal_xyz_fn(point1,point2,point3)
cylindrical_unit_vectors=calculate_cylindrical_unit_vectors(point1)
surface_normal_rTz=cylindrical_transform_surface_normal_xyz(surface_normal_xyz,cylindrical_unit_vectors,tile)
vector_data=create_vector_data_cyl(surface_normal_rTz, cylindrical_unit_vectors, point1)
tilt_angle=toroidal_tilt_wrt_xy(surface_normal_xyz, cylindrical_unit_vectors)
print(tilt_angle)

rz_normal=tile_unit_components_rz(surface_normal_rTz)
probe_angle_wrt_rz_normal=10*np.pi/180
scaling=np.cos(probe_angle_wrt_rz_normal)/(rz_normal[0]**2+rz_normal[2]**2)
probe_normal_theta=(1-(scaling**2)*(rz_normal[0]**2+rz_normal[2]**2))**0.5

loc='lower'

if loc=='upper':
    probe_surface_normal_rTz=np.array([scaling*rz_normal[0] , probe_normal_theta ,-1*scaling*rz_normal[2]  ])
if loc=='lower':
    probe_surface_normal_rTz=np.array([scaling*rz_normal[0] , probe_normal_theta , scaling*rz_normal[2]  ])

mag_probe_surface_normal_rtz=(probe_surface_normal_rTz[0]**2+probe_surface_normal_rTz[1]**2+probe_surface_normal_rTz[2]**2)**0.5
print(mag_probe_surface_normal_rtz)
vector_data_probe=create_vector_data_cyl(probe_surface_normal_rTz, cylindrical_unit_vectors, point1)
######################tile shifts for probe surface normals

#can do affine transforms for several probes at once.. repeated same data in example below.
probe_surface_normal_r_unshifted=np.array([2,1])*probe_surface_normal_rTz[0]
probe_surface_normal_z_unshifted=np.array([2,1])*probe_surface_normal_rTz[2]
tile_array=np.array([[tile],[tile]])
loc_array=np.array([[loc],[loc]])

probe_surface_normal_r_shifted, probe_surface_normal_z_shifted=RZ_shifts(probe_surface_normal_r_unshifted,probe_surface_normal_z_unshifted,tile_array,loc_array)
mag_rz_unshifted=(probe_surface_normal_rTz[0]**2+probe_surface_normal_rTz[2]**2)**0.5

scaling_RZ=mag_rz_unshifted/(probe_surface_normal_r_shifted[0][0]**2+probe_surface_normal_z_shifted[0][0]**2)**0.5
probe_surface_normal_rTz_shifted=np.array([scaling_RZ*probe_surface_normal_r_shifted[0][0], probe_surface_normal_rTz[1] ,scaling_RZ*probe_surface_normal_z_shifted[0][0] ])

mag=(probe_surface_normal_rTz_shifted[0]**2+probe_surface_normal_rTz_shifted[1]**2+probe_surface_normal_rTz_shifted[2]**2)**0.5
print(mag)





















radius_surface, theta_surface, z_surface=cartesian_to_cylindrical_position_coordinates(filename)
plt.figure()
plt.scatter(radius_surface,z_surface)
plt.figure()
plt.scatter(radius_surface,theta_surface)
plt.show()



fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(len(x_data_rows)):
    plt.plot(x_data_rows[i],y_data_rows[i],z_data_rows[i],'o',linestyle='')
plt.plot(vector_data[0],vector_data[1],vector_data[2])
plt.plot(vector_data_probe[0],vector_data_probe[1],vector_data_probe[2],'r')
plt.show()

print(surface_normal_xyz)