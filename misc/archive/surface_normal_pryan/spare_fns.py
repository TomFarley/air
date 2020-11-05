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


def cartesian_to_cylindrical_position_coordinates(x_data_rows, y_data_rows, z_data_rows):
    """
    convert position coordinates in cartesian coordinates to cylindrical coordinates.
    """
        surface=pd.read_csv(filename)
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
