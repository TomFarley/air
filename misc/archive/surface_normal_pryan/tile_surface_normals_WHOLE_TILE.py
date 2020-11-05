import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import pyuda
client=pyuda.Client()

def tile_rows(filename):
    """
    filename is the name given to the Fishpool csv file. This file must have the following column headings: 'x', 'y' and 'z'.
    This function organises the csv data into 'rows' and 'columns'. The row data is represented as a list, and the elements within a list are np.arrays (column data).
    There are three outputs; one for each basis vector in cartesian coordinates.
    """
    surface=pd.read_csv(filename)
    x_surface=np.array(surface['x'])
    y_surface=np.array(surface['y'])
    z_surface=np.array(surface['z'])

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
    """
    filename is the name given to the Fishpool csv file. This file must have the following column headings: 'x', 'y' and 'z'.
    This function organises the csv data into 'rows' and 'columns'. The row data is represented as a list, and the elements within a list are np.arrays (column data).
    In addition, the function crops the csv data so that only data between min_theta and max_theta is kept (cylindrical coordinates).
    There are three outputs; one for each basis vector in cartesian coordinates.
    """
    surface=pd.read_csv(filename)
    surface=pd.read_csv(filename)
    x_surface_uncut=np.array(surface['x'])
    y_surface_uncut=np.array(surface['y'])
    z_surface_uncut=np.array(surface['z'])

    radius_surface=(x_surface_uncut**2+y_surface_uncut**2)**0.5
    theta_surface=np.arcsin(y_surface_uncut/radius_surface)  # TF: NOTE using arcsin not arctan

    x_surface=np.array([]) #these will define the surface after cropping the data.
    y_surface=np.array([])
    z_surface=np.array([])

    for i in range(len(x_surface_uncut)):
        if (theta_surface[i]>min_theta-1e-4) & (theta_surface[i]<max_theta+1e-4): #added a small constant in case theta_surface has rounding errors.
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
    # NOTE: Each row is a radial strip of 80 points
    return x_data_rows, y_data_rows, z_data_rows  # QUESTION: Arrays all have 80 elements so could cast list to ndarray?


def three_points(x_data_rows, y_data_rows, z_data_rows, row_index, column_index):
    """
    Locate three, closely separated, points on the surface.
    One of these points is defined by row_index, column_index; called point1.
    The other two points are have a common row, which is a different row to point1
    Outputs are the three points in cartesian coordinates.
    """
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

def calculate_surface_normal_xyz(point1,point2,point3):
    """
    Cross product to calculate surface normal using the three points. Inputs and outputs are in cartesian coordinates.
    """
    surface_vector1=point1-point2
    surface_vector2=point1-point3
    one_cross_two=np.cross(surface_vector2,surface_vector1)
    magnitude=(one_cross_two[0]**2+one_cross_two[1]**2+one_cross_two[2]**2)**0.5
    surface_normal_xyz=one_cross_two/magnitude
    return surface_normal_xyz


def tile_unit_components_rz(surface_normal_rTz):
    """
    Take a surface normal in cylindrical coordinates, set the theta component to zero, and renormalise.
    """
    rz_normal_cylind=np.array([surface_normal_rTz[0],0,surface_normal_rTz[2]])/(surface_normal_rTz[0]**2+surface_normal_rTz[2]**2)**0.5
    return rz_normal_cylind

def calculate_cylindrical_unit_vectors(point1):
    """
    Calculate cylindrical unit vectors at position point1.
    point1 is given in cartesian coordinates.
    Output is the cylindrical unit vectors written in terms of cartesian coordinates.
    """
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


def transform_vector_into_cylindrical(surface_normal_xyz,cylindrical_unit_vectors):
    """
    Transform a vector given in terms of cartesian coordinates to cylindrical coordinates.
    """
    radial_unit_vector=cylindrical_unit_vectors[0]
    toroidal_unit_vector=cylindrical_unit_vectors[1]
    if np.abs(radial_unit_vector[0])>1e-10: #problem with infinity when the radial unit vector has a small x component.
        # TF: T = (Y - (R_y * X / R_x)) / (T_y - (T_x * R_y / R_x))
        surface_normal_toroidal=(surface_normal_xyz[1]-(radial_unit_vector[1]*surface_normal_xyz[0]/radial_unit_vector[0]))/(toroidal_unit_vector[1]-(toroidal_unit_vector[0]*radial_unit_vector[1]/radial_unit_vector[0]))
        surface_normal_radial=(surface_normal_xyz[0]-(surface_normal_toroidal*toroidal_unit_vector[0]))/radial_unit_vector[0]
    else:
        surface_normal_toroidal=surface_normal_xyz[0]/toroidal_unit_vector[0]
        surface_normal_radial=(surface_normal_xyz[1]-(toroidal_unit_vector[1]*surface_normal_toroidal)/radial_unit_vector[1])
    surface_normal_rTz=np.array([surface_normal_radial,surface_normal_toroidal,surface_normal_xyz[2]]) #equivalent to surface_normal_xyz but in new basis.
    magnitude_rTz=(surface_normal_rTz[0]**2+surface_normal_rTz[1]**2+surface_normal_rTz[2]**2)**0.5 #check
    return surface_normal_rTz



def ensure_outwards_normal(surface_normal_rTz,cylindrical_unit_vectors,point1,loc,tile):
    """
    ensure that the surface normals point outwards from the surface.
    The code generates inwards and outwards surface normal-vectors at position point1. Distance to ref position is used to determine the outwards vector.
    Ref position is dependent on the type of tile.
    """
    if (tile=='N1') or (tile=='N2') or (tile=='B1') or (tile=='B2') or (tile=='B3') or (tile=='B4') or (tile=='B5'):
        ref_pos=np.array([1.25,0,-1.4]) #outwards should point away from this point.
    else:
        ref_pos=np.array([1.2,0,-1.5]) #outwards should point towards this point. e.g. tile 'T2'
    vector_xyz=create_vector_cartesian(surface_normal_rTz, cylindrical_unit_vectors, point1,1)
    opp_vector_xyz=create_vector_cartesian(surface_normal_rTz, cylindrical_unit_vectors, point1,-1)
    distance=((vector_xyz[0][1]-ref_pos[0])**2+(vector_xyz[1][1])**2+(vector_xyz[2][1]-ref_pos[2])**2)**0.5
    opp_distance=((opp_vector_xyz[0][1]-ref_pos[0])**2+(opp_vector_xyz[1][1])**2+(opp_vector_xyz[2][1]-ref_pos[2])**2)**0.5
    if (tile=='N1') or (tile=='N2') or (tile=='B1') or (tile=='B2') or (tile=='B3') or (tile=='B4') or (tile=='B5'):
        if distance>opp_distance:
            dir_surface_normal_rTz=surface_normal_rTz
        else:
            dir_surface_normal_rTz=surface_normal_rTz*-1
    else:
        if distance<opp_distance:
            dir_surface_normal_rTz=surface_normal_rTz
        else:
            dir_surface_normal_rTz=surface_normal_rTz*-1
    if loc=='upper':
        dir_surface_normal_rTz[2]=dir_surface_normal_rTz[2]*-1
    return dir_surface_normal_rTz

def create_vector_cartesian(surface_normal_rTz, cylindrical_unit_vectors, point1,polarity): #polarity controls direct of surface normal but not position at which it is drawn from.
    """
    Generates data points for plotting surface-normals in cartesian coordinates.
    surface_normal_rTz in cylindrical coords.
    cylindrical_unit_vectors and point1 in cartesian.
    """
    radial_unit_vector=cylindrical_unit_vectors[0]
    toroidal_unit_vector=cylindrical_unit_vectors[1]
    z_unit_vector=cylindrical_unit_vectors[2]
    surface_normal_xyz=surface_normal_rTz[0]*radial_unit_vector+surface_normal_rTz[1]*toroidal_unit_vector+surface_normal_rTz[2]*z_unit_vector
    x_points=np.array([0,polarity*surface_normal_xyz[0]])+point1[0]
    y_points=np.array([0,polarity*surface_normal_xyz[1]])+point1[1]
    z_points=np.array([0,polarity*surface_normal_xyz[2]])+point1[2]
    vector_xyz=[x_points, y_points, z_points]
    return vector_xyz

def generate_surface_norms(tile,min_theta,max_theta, row_resolution, column_resolution):
    """
    main function - calls the other functions. Generates surface-normals across the entire surface defined in the csv file. This is only for LOWER.
    row_resolution (radial), column_resolution (toroidal) in metres.
    tile is e.g. 'T2'
    min_theta,max_theta for cropping the Fishpool csv data.
    """
    filename=tile+'.csv'
    x_data_rows, y_data_rows, z_data_rows=tile_rows_cut(filename,min_theta,max_theta)
    row_step=((x_data_rows[1][0]-x_data_rows[0][0])**2+(y_data_rows[1][0]-y_data_rows[0][0])**2+(z_data_rows[1][0]-z_data_rows[0][0])**2)**0.5 #unit m
    column_step=((x_data_rows[0][1]-x_data_rows[0][0])**2+(y_data_rows[0][1]-y_data_rows[0][0])**2+(z_data_rows[0][1]-z_data_rows[0][0])**2)**0.5 #unit m
    if row_step>row_resolution:
        print('Error! Row_resolution is smaller than the resolution of the data.')
        row_resolution=row_step
    if column_step>column_resolution:
        print('Error! Column_resolution is smaller than the resolution of the data.')
        column_resolution=column_step
    row_resolution_i=np.floor(row_resolution/row_step)
    column_resolution_i=np.floor(column_resolution/column_step)

    row_indices=np.arange(0,len(x_data_rows),row_resolution_i)
    row_indices=row_indices.astype(int)
    count=0
    for row in row_indices:
        column_indices=np.arange(0,len(x_data_rows[row]),column_resolution_i)
        column_indices=column_indices.astype(int)
        for col in column_indices:
            point1,point2,point3=three_points(x_data_rows, y_data_rows, z_data_rows, row, col) #points on a tile surface z<0
            surface_normal_xyz=calculate_surface_normal_xyz(point1,point2,point3)
            cylindrical_unit_vectors=calculate_cylindrical_unit_vectors(point1)
            surface_normal_rTz=transform_vector_into_cylindrical(surface_normal_xyz,cylindrical_unit_vectors) #surface-normal might not point outwards from surface
            dir_surface_normal_rTz=ensure_outwards_normal(surface_normal_rTz,cylindrical_unit_vectors,point1,'lower',tile) #surface-normal points outwards.
            dir_surface_normal_rTz=dir_surface_normal_rTz.reshape(1,3)
            point1=point1.reshape(1,3)
            if count==0:
                store_vectors_rTz=dir_surface_normal_rTz
                store_positions_xyz=point1
            else:
                store_vectors_rTz=np.concatenate([store_vectors_rTz,dir_surface_normal_rTz])
                store_positions_xyz=np.concatenate([store_positions_xyz,point1])
            count=count+1
    return store_positions_xyz,store_vectors_rTz,x_data_rows,y_data_rows,z_data_rows

def plot_surface_norms(store_positions_xyz,store_vectors_rTz,x_data_rows,y_data_rows,z_data_rows):
    """
    3D plot of surface normal vectors
    """
    fig=plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(len(x_data_rows)):
        plt.plot(x_data_rows[i],y_data_rows[i],z_data_rows[i],'o',linestyle='',color='k')
    for j in range(len(store_positions_xyz)):
        cylindrical_unit_vectors=calculate_cylindrical_unit_vectors(store_positions_xyz[j])
        outward_vector_xyz=create_vector_cartesian(store_vectors_rTz[j], cylindrical_unit_vectors, store_positions_xyz[j],1) #need to input correct position.. in this situation always input 1 rather than -1.
        plt.plot(outward_vector_xyz[0],outward_vector_xyz[1],outward_vector_xyz[2])
    plt.show()
    return

def transform_store_positions_into_cylind(store_positions_xyz):
    """
    transform store_position_xyz into cylindrical coordinates.
    """
    for i in range(len(store_positions_xyz)):
        radius_xy=(store_positions_xyz[i][0]**2+store_positions_xyz[i][1]**2)**0.5
        theta_xy=2*np.pi-np.abs((np.arcsin(store_positions_xyz[i][1]/radius_xy)))
        z_component=store_positions_xyz[i][2]
        element=np.array([radius_xy,theta_xy,z_component]).reshape(1,3)
        if i==0:
            store_positions_rTz=element
        else:
            store_positions_rTz=np.concatenate([store_positions_rTz,element])
    return store_positions_rTz


def lookup_surface_normal_near_probe(large_or_small_theta_edge,probe_position_rTz,store_positions_xyz,store_vectors_rTz,store_positions_rTz=None):
    """
    Locate the tile surface-normal that is closest to a probe.
    Probes are located at tile boundaries (e.g. T2-T2 but not T2-T3).
    large_or_small_theta_edge='large' would take surface-normal of tile from the largest theta values.
    large_or_small_theta_edge='small' would take surface-normal of tile from the smallest theta values.
    At extremes of theta, expect difference in 'height' of tile due to toroidal shadowing.
    The output is dependent on z polarity in position of probe.
    """
    if store_positions_rTz is None:
        store_positions_rTz=transform_store_positions_into_cylind(store_positions_xyz)
    if large_or_small_theta_edge=='large':
        max_theta=max(store_positions_rTz[:,1])
        indices_interest,=np.where(store_positions_rTz[:,1]>max_theta-1e-4)
    if large_or_small_theta_edge=='small':
        min_theta=min(store_positions_rTz[:,1])
        indices_interest,=np.where(store_positions_rTz[:,1]<min_theta+1e-4)
    interest_positions=store_positions_rTz[indices_interest,:]
    interest_vectors=store_vectors_rTz[indices_interest,:]
    if probe_position_rTz[2]>=0: #probes located UPPER
        z_pol=-1
    else:
        z_pol=1 #probes located LOWER
    probe_z=probe_position_rTz[2]*z_pol #should always be negative
    rz_diff=((probe_position_rTz[0]-interest_positions[:,0])**2+(probe_z-interest_positions[:,2])**2)**0.5
    final_index,=np.where(rz_diff==min(rz_diff))[0]
    probe_tile_SN_rTz=np.array([interest_vectors[final_index,0],interest_vectors[final_index,1],interest_vectors[final_index,2]*z_pol]).reshape(1,3)
    probe_csvtile_position_rTz=np.array([interest_positions[final_index,0],interest_positions[final_index,1],interest_positions[final_index,2]*z_pol]).reshape(1,3)
    return probe_tile_SN_rTz,probe_csvtile_position_rTz

if __name__ == '__main__':
    tile='T4'
    max_theta=0
    min_theta=-15*np.pi/180
    row_resolution=20e-3 #units of m
    column_resolution=20e-3 #units of m


    store_positions_xyz,store_vectors_rTz,x_data_rows,y_data_rows,z_data_rows=generate_surface_norms(tile,min_theta,max_theta, row_resolution, column_resolution)
    plot_surface_norms(store_positions_xyz,store_vectors_rTz,x_data_rows,y_data_rows,z_data_rows)

    probe_position_rTz=np.array([1.2,0,2.06]) #T4 probe
    large_or_small_theta_edge='large'
    probe_tile_SN_rTz,probe_csvtile_position_rTz=lookup_surface_normal_near_probe(large_or_small_theta_edge,probe_position_rTz,store_positions_xyz,store_vectors_rTz,store_positions_rTz=None)
    #theta value in probe_csvtile_position_rTz is the theta value from the csv file; not for the probe position!
    #probe_csvtile_position_rTz will have the same z polarity as the probe position.
    print(probe_tile_SN_rTz)
    print(probe_csvtile_position_rTz)

    #of interest to LP's
    tile='T5A'
    min_theta=-7.5*np.pi/180
    max_theta=0

    tile='T5D'
    min_theta=-30*np.pi/180
    max_theta=-22.5*np.pi/180
