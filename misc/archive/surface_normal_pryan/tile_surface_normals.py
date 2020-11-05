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


def tile_rows_cut(filename,min_theta,max_theta):
    surface=pd.read_csv(filename)
    x_surface_uncut=np.array(surface['x'],dtype = np.float32)
    y_surface_uncut=np.array(surface['y'],dtype = np.float32)
    z_surface_uncut=np.array(surface['z'],dtype = np.float32)

    radius_surface=(x_surface_uncut**2+y_surface_uncut**2)**0.5
    theta_surface=np.arcsin(y_surface_uncut/radius_surface)

    x_surface=np.array([])
    y_surface=np.array([])
    z_surface=np.array([])

    for i in range(len(x_surface_uncut)):
        if (theta_surface[i]>=min_theta-1e-4) & (theta_surface[i]<=max_theta+1e-4):
            x_surface=np.concatenate([x_surface,np.array([x_surface_uncut[i]])])
            y_surface=np.concatenate([y_surface,np.array([y_surface_uncut[i]])])
            z_surface=np.concatenate([z_surface,np.array([z_surface_uncut[i]])])

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


def cartesian_to_cylindrical_position_coordinates(x_data_rows, y_data_rows, z_data_rows):
    radius_data_rows=[]
    theta_data_rows=[]
    for i in range(len(x_data_rows)):
        x_data_i=x_data_rows[i]
        y_data_i=y_data_rows[i]
        radius_data_i=(x_data_i**2+y_data_i**2)**0.5
        theta_data_i=np.arcsin(y_data_i/radius_data_i)
        radius_data_rows.append(radius_data_i)
        theta_data_rows.append(theta_data_i)
    return radius_data_rows, theta_data_rows,z_data_rows

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

def average_surface_normals(x_data_rows, y_data_rows, z_data_rows,column_index):
    all_surface_normal_xyz=[]
    sum_x_component=0
    sum_y_component=0
    sum_z_component=0
    all_surface_normal_rTz=[]
    sum_r_component=0
    sum_theta_component=0
    sum_z_component=0
    all_toroidal_angle=[]
    for i in range(len(x_data_rows)):
        #column index as 0 or -1 for LP positions
        point1,point2,point3=three_points(x_data_rows, y_data_rows, z_data_rows, i, column_index)
        surface_normal_xyz=surface_normal_xyz_fn(point1,point2,point3)
        if surface_normal_xyz[2]<0:
            surface_normal_xyz=surface_normal_xyz*-1 #make sure all have same polarity before averaging
        all_surface_normal_xyz.append(surface_normal_xyz)
        sum_x_component=surface_normal_xyz[0]+sum_x_component
        sum_y_component=surface_normal_xyz[1]+sum_y_component
        sum_z_component=surface_normal_xyz[2]+sum_z_component
        mean_surface_normal_xyz=np.array([sum_x_component,sum_y_component,sum_z_component])/((sum_x_component**2+sum_y_component**2+sum_z_component**2)**0.5)

        cylindrical_unit_vectors=calculate_cylindrical_unit_vectors(point1)
        surface_normal_rTz=calculate_SN_cylind(surface_normal_xyz,cylindrical_unit_vectors)

        rz_normal_cylind=tile_unit_components_rz(surface_normal_rTz)

        toroidal_angle=np.arccos(np.dot(surface_normal_rTz,rz_normal_cylind))*180/np.pi
        all_toroidal_angle.append(toroidal_angle)

        all_surface_normal_rTz.append(surface_normal_rTz)
        sum_r_component=surface_normal_rTz[0]+sum_r_component
        sum_theta_component=surface_normal_rTz[1]+sum_theta_component
        mean_surface_normal_rTz=np.array([sum_r_component,sum_theta_component,sum_z_component])/((sum_r_component**2+sum_theta_component**2+sum_z_component**2)**0.5)
    return mean_surface_normal_xyz, all_surface_normal_xyz, mean_surface_normal_rTz, all_surface_normal_rTz, all_toroidal_angle


def tile_unit_components_rz(surface_normal_rTz):
    rz_normal_cylind=np.array([surface_normal_rTz[0],0,surface_normal_rTz[2]])/(surface_normal_rTz[0]**2+surface_normal_rTz[2]**2)**0.5
    return rz_normal_cylind

def calculate_cylindrical_unit_vectors(point1):
    radial_unit_vector=np.array([point1[0],point1[1],0])/((point1[0]**2+point1[1]**2)**0.5)

    toroidal_unit_vector_magnitude_y=np.abs(radial_unit_vector[0]/(radial_unit_vector[0]**2+radial_unit_vector[1]**2)**0.5)
    toroidal_unit_vector_magnitude_x=np.abs((1-toroidal_unit_vector_magnitude_y**2)**0.5)

    if (point1[0]>0) & (point1[1]>=0):
        toroidal_unit_vector=np.array([-toroidal_unit_vector_magnitude_x,toroidal_unit_vector_magnitude_y,0])
    if (point1[0]<=0) & (point1[1]>0):
        toroidal_unit_vector=np.array([-toroidal_unit_vector_magnitude_x,-toroidal_unit_vector_magnitude_y,0])
    if (point1[0]<0) & (point1[1]<=0):
        toroidal_unit_vector=np.array([toroidal_unit_vector_magnitude_x,-toroidal_unit_vector_magnitude_y,0])
    if (point1[0]>=0) & (point1[1]<0):
        toroidal_unit_vector=np.array([toroidal_unit_vector_magnitude_x,toroidal_unit_vector_magnitude_y,0])

    z_unit_vector=np.array([0,0,1])
    cylindrical_unit_vectors=np.array([radial_unit_vector, toroidal_unit_vector, z_unit_vector])
    return cylindrical_unit_vectors


def calculate_SN_cylind(surface_normal_xyz,cylindrical_unit_vectors):
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
    return surface_normal_rTz

def locate_nearest_tile_normal(x_data_rows, y_data_rows, z_data_rows,probe_z,column_index):
    if probe_z>=0:
        probe_z=probe_z*-1
    endx=[]
    endy=[]
    endz=[]
    [endz.append(z_data_rows[i][column_index]) for i in range(len(z_data_rows))]
    endz_np=np.array(endz)
    diff=np.abs(endz_np-probe_z)
    index,=np.where(diff==min(diff))
    row_index=index[column_index]
    column_index=0
    return row_index, column_index

def ensure_outwards_normal(surface_normal_rTz,cylindrical_unit_vectors,point1,loc,tile):
    if tile==('N1' or 'N2' or 'B1' or 'B2' or 'B3' or 'B4' or 'B5'):
        ref_pos=np.array([1.25,0,-1.3])
    else:
        ref_pos=np.array([1.2,0,-1.5])
    vector_xyz=create_vector_cartesian(surface_normal_rTz, cylindrical_unit_vectors, point1,1)
    opp_vector_xyz=create_vector_cartesian(surface_normal_rTz, cylindrical_unit_vectors, point1,-1)
    distance=((vector_xyz[0][1]-ref_pos[0])**2+(vector_xyz[1][1])**2+(vector_xyz[2][1]-ref_pos[2])**2)**0.5
    opp_distance=((opp_vector_xyz[0][1]-ref_pos[0])**2+(opp_vector_xyz[1][1])**2+(opp_vector_xyz[2][1]-ref_pos[2])**2)**0.5
    if distance<opp_distance:
        dir_surface_normal_rTz=surface_normal_rTz
    else:
        dir_surface_normal_rTz=surface_normal_rTz.copy()*-1
    if loc=='upper':
        dir_surface_normal_rTz[2]=dir_surface_normal_rTz[2].copy()*-1
    return dir_surface_normal_rTz

def create_vector_cartesian(surface_normal_rTz, cylindrical_unit_vectors, point1,polarity): #polarity controls direct of surface normal but not position at which it is drawn from.
    radial_unit_vector=cylindrical_unit_vectors[0]
    toroidal_unit_vector=cylindrical_unit_vectors[1]
    z_unit_vector=cylindrical_unit_vectors[2]
    surface_normal_xyz=surface_normal_rTz[0]*radial_unit_vector+surface_normal_rTz[1]*toroidal_unit_vector+surface_normal_rTz[2]*z_unit_vector
    x_points=np.array([0,polarity*surface_normal_xyz[0]])+point1[0]
    y_points=np.array([0,polarity*surface_normal_xyz[1]])+point1[1]
    z_points=np.array([0,polarity*surface_normal_xyz[2]])+point1[2]
    vector_xyz=[x_points, y_points, z_points]
    return vector_xyz


probe_z=2
if probe_z<0:
    loc='lower'
    loc_p=1
else:
    loc='upper'
    loc_p=-1

tile='T4'
filename=tile+'.csv'
max_theta=0
min_theta=-15*np.pi/180
x_data_rows, y_data_rows, z_data_rows=tile_rows_cut(filename,min_theta,max_theta)
column_index=0


row_index,column_index=locate_nearest_tile_normal(x_data_rows, y_data_rows, z_data_rows,probe_z,column_index) #for probe with z>0, converts to z<0
point1,point2,point3=three_points(x_data_rows, y_data_rows, z_data_rows, row_index, column_index) #points on a tile surface z<0
surface_normal_xyz=surface_normal_xyz_fn(point1,point2,point3)
cylindrical_unit_vectors=calculate_cylindrical_unit_vectors(point1) #unit vectors independent of probe z polarity.
surface_normal_rTz=calculate_SN_cylind(surface_normal_xyz,cylindrical_unit_vectors) #will give the wrong z polarity for z>0.
dir_surface_normal_rTz=ensure_outwards_normal(surface_normal_rTz,cylindrical_unit_vectors,point1,loc,tile) #gives correct direction for any z polarity.




outward_vector=create_vector_cartesian(dir_surface_normal_rTz, cylindrical_unit_vectors, np.array([point1[0],point1[1],point1[2]*loc_p]),1) #need to input correct position.. in this situation always input 1 rather than -1.
fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(len(x_data_rows)):
    plt.plot(x_data_rows[i],y_data_rows[i],loc_p*z_data_rows[i],'o',linestyle='')
plt.plot(outward_vector[0],outward_vector[1],outward_vector[2])
plt.show()






#mean_surface_normal_xyz, all_surface_normal_xyz, mean_surface_normal_rTz, all_surface_normal_rTz, all_toroidal_angle=average_surface_normals(x_data_rows, y_data_rows, z_data_rows,0)
