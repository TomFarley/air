import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd


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


def cylindrical_transform_surface_normal_xyz(surface_normal_xyz,cylindrical_unit_vectors):
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


filename='T4.csv'
x_data_rows, y_data_rows, z_data_rows=tile_rows(filename)
point1, point2, point3=three_points(x_data_rows, y_data_rows, z_data_rows, 0,0)
surface_normal_xyz=surface_normal_xyz_fn(point1,point2,point3)
cylindrical_unit_vectors=calculate_cylindrical_unit_vectors(point1)
surface_normal_rTz=cylindrical_transform_surface_normal_xyz(surface_normal_xyz,cylindrical_unit_vectors)
vector_data=create_vector_data_cyl(surface_normal_rTz, cylindrical_unit_vectors, point1)
tilt_angle=toroidal_tilt_wrt_xy(surface_normal_xyz, cylindrical_unit_vectors)
print(tilt_angle)

x_data_rows, y_data_rows, z_data_rows=tile_rows(filename)
point1_2, point2_2, point3_2=three_points(x_data_rows, y_data_rows, z_data_rows, 0,55)
surface_normal_xyz2=surface_normal_xyz_fn(point1_2,point2_2,point3_2)
cylindrical_unit_vectors2=calculate_cylindrical_unit_vectors(point1_2)
surface_normal_rTz2=cylindrical_transform_surface_normal_xyz(surface_normal_xyz,cylindrical_unit_vectors)
vector_data2=create_vector_data_cyl(surface_normal_rTz, cylindrical_unit_vectors, point1_2)

x_data_rows, y_data_rows, z_data_rows=tile_rows(filename)
point1_3, point2_3, point3_3=three_points(x_data_rows, y_data_rows, z_data_rows, 40,55)
surface_normal_xyz3=surface_normal_xyz_fn(point1_3,point2_3,point3_3)
cylindrical_unit_vectors3=calculate_cylindrical_unit_vectors(point1_3)
surface_normal_rTz3=cylindrical_transform_surface_normal_xyz(surface_normal_xyz,cylindrical_unit_vectors)
vector_data3=create_vector_data_cyl(surface_normal_rTz, cylindrical_unit_vectors, point1_3)

x_data_rows, y_data_rows, z_data_rows=tile_rows(filename)
point1_4, point2_4, point3_4=three_points(x_data_rows, y_data_rows, z_data_rows, 60,0)
surface_normal_xyz4=surface_normal_xyz_fn(point1_4,point2_4,point3_4)
cylindrical_unit_vectors4=calculate_cylindrical_unit_vectors(point1_4)
surface_normal_rTz4=cylindrical_transform_surface_normal_xyz(surface_normal_xyz,cylindrical_unit_vectors)
vector_data4=create_vector_data_cyl(surface_normal_rTz, cylindrical_unit_vectors, point1_4)

combine_point=np.array([(point1[0],point2[0],point3[0]), (point1[1],point2[1],point3[1]), (point1[2],point2[2],point3[2])  ])
combine_point2=np.array([(point1_2[0],point2_2[0],point3_2[0]), (point1_2[1],point2_2[1],point3_2[1]), (point1_2[2],point2_2[2],point3_2[2])  ])
combine_point3=np.array([(point1_3[0],point2_3[0],point3_3[0]), (point1_3[1],point2_3[1],point3_3[1]), (point1_3[2],point2_3[2],point3_3[2])  ])
combine_point4=np.array([(point1_4[0],point2_4[0],point3_4[0]), (point1_4[1],point2_4[1],point3_4[1]), (point1_4[2],point2_4[2],point3_4[2])  ])

fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(len(x_data_rows)):
    plt.plot(x_data_rows[i],y_data_rows[i],z_data_rows[i],'o',linestyle='',color='k')
plt.plot(combine_point[0],combine_point[1],combine_point[2],'o',linestyle='',color='r')
plt.plot(combine_point2[0],combine_point2[1],combine_point2[2],'o',linestyle='',color='r')
plt.plot(combine_point3[0],combine_point3[1],combine_point3[2],'o',linestyle='',color='r')
plt.plot(combine_point4[0],combine_point4[1],combine_point4[2],'o',linestyle='',color='r')
plt.plot(vector_data[0],vector_data[1],vector_data[2])
plt.plot(vector_data2[0],vector_data2[1],vector_data2[2])
plt.plot(vector_data3[0],vector_data3[1],vector_data3[2])
plt.plot(vector_data4[0],vector_data4[1],vector_data4[2])
plt.show()
