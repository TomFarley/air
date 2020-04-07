# Peter Ryan March 2020 - MAST-U 's' coordinate script used as reference for tools in FIRE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyuda
client=pyuda.Client()

limiter_noshift =client.geometry("/limiter/efit",50000, no_cal=True)


##################################################

r_div_noshift1=limiter_noshift.data.R
z_div_noshift1=limiter_noshift.data.Z

#########################################################

#the line does not start at a convenient place - traces from upper to lower and then back to upper (anticlockwise)
#this section of the code shifts the line so that the ENTIRE of upper is traced first, and then lower is traced.

index_lower=np.where(z_div_noshift1<0) #lower coords
r_div_noshift_l=r_div_noshift1[index_lower]
z_div_noshift_l=z_div_noshift1[index_lower]

index_upp=np.where(z_div_noshift1>0) #upper coords
r_div_noshift_u=r_div_noshift1[index_upp]
z_div_noshift_u=z_div_noshift1[index_upp]

r_start_u=r_div_noshift_u[16::] #this was an easy fix to give start and end points of trace - not ideal.
z_start_u=z_div_noshift_u[16::]
r_end_u=r_div_noshift_u[0:16]
z_end_u=z_div_noshift_u[0:16]

r_div_noshift=np.concatenate([r_start_u,r_end_u,r_div_noshift_l])
z_div_noshift=np.concatenate([z_start_u,z_end_u,z_div_noshift_l])
#plot the following:
#plt.figure()
#plt.plot(r_div_noshift,z_div_noshift)
#plt.plot(r_div_noshift[0:15],z_div_noshift[0:15])
#plt.show()



def create_R_Z_s_coords(r_div,z_div):
#input trace of machine surface and output R, Z, s for surface. The input trace must start at upper and be anticlockwise. Remember RZ inputs define the start and end points of straight lines and not coordinates for along the line!
    for i in range(len(r_div)-1):
        total_delta_R=r_div[i+1]-r_div[i]
        total_delta_Z=z_div[i+1]-z_div[i]
        length_component=(total_delta_R**2+total_delta_Z**2)**0.5 # distance between two adjacent points using r_div and z_div.
        angle=np.arctan(total_delta_Z/total_delta_R)
        partial_s=np.arange(0,length_component,0.0001) #small steps along 'length_component'
        delta_R=np.abs(partial_s*np.cos(angle)) #R coords along the line
        delta_Z=np.abs(partial_s*np.sin(angle)) #Z_coords along the line
        if i==0: #use the first line in RZ to setup the coordinate vectors; these will be extended by concatenation.
            s_coord=partial_s.copy()
            if r_div[1]>r_div[0]:
                R_coord=delta_R+r_div[0]
            else:
                R_coord=-delta_R+r_div[0]
            if z_div[1]>z_div[0]:
                Z_coord=delta_Z+z_div[0]
            else:
                Z_coord=-delta_Z+z_div[0]
        else:
            current_max_s=max(s_coord)
            s_coord=np.concatenate((s_coord,current_max_s+partial_s),axis=0)
            if r_div[i+1]>r_div[i]:
                R_coord=np.concatenate((R_coord,delta_R+r_div[i]),axis=0)
            else:
                R_coord=np.concatenate((R_coord,-delta_R+r_div[i]),axis=0)
            if z_div[i+1]>z_div[i]:
                Z_coord=np.concatenate((Z_coord,delta_Z+z_div[i]),axis=0)
            else:
                Z_coord=np.concatenate((Z_coord,-delta_Z+z_div[i]),axis=0)

############################################################
    #locate Z where s=0.. This works if start point is Z>0 and div surface drawn anticlockwise.
    index_lower=np.where(Z_coord<0)
    R_div_lower=R_coord[index_lower]
    Z_div_lower=Z_coord[index_lower]
    s_div_lower1=s_coord[index_lower]

    s_offset=s_div_lower1[0]
    s_div_lower=s_div_lower1-s_offset

############################################################
    index_upper=np.where(Z_coord>0)
    R_div_upper=R_coord[index_upper]
    Z_div_upper=Z_coord[index_upper]
    s_div_upper1=s_coord[index_upper]

    s_offset=s_div_upper1[-1]
    s_div_upper=np.abs(s_div_upper1-s_offset)

##########################################################

    R_coord_lower_T=np.transpose(np.ones((1,len(R_div_lower)))*R_div_lower)
    Z_coord_lower_T=np.transpose(np.ones((1,len(Z_div_lower)))*Z_div_lower)
    s_coord_lower_T=np.transpose(np.ones((1,len(s_div_lower)))*s_div_lower)
    R_coord_upper_T=np.transpose(np.ones((1,len(R_div_upper)))*R_div_upper)
    Z_coord_upper_T=np.transpose(np.ones((1,len(Z_div_upper)))*Z_div_upper)
    s_coord_upper_T=np.transpose(np.ones((1,len(s_div_upper)))*s_div_upper)

    matrix_coord_data_upper=np.concatenate((R_coord_upper_T,Z_coord_upper_T,s_coord_upper_T),axis=1)
    matrix_DIV_coord_upper=pd.DataFrame(matrix_coord_data_upper,columns=['R_upper','Z_upper','s_upper'])

    matrix_coord_data_lower=np.concatenate((R_coord_lower_T,Z_coord_lower_T,s_coord_lower_T),axis=1)
    matrix_DIV_coord_lower=pd.DataFrame(matrix_coord_data_lower,columns=['R_lower','Z_lower','s_lower'])

    return matrix_DIV_coord_lower,matrix_DIV_coord_upper



def locate_probe_s(R_probe,Z_probe,matrix_DIV_coord_lower,matrix_DIV_coord_upper):
    #give R and Z, output s value.
    s_coord_lower=np.array(matrix_DIV_coord_lower["s_lower"])
    R_coord_lower=np.array(matrix_DIV_coord_lower["R_lower"])
    Z_coord_lower=np.array(matrix_DIV_coord_lower["Z_lower"])
    s_coord_upper=np.array(matrix_DIV_coord_upper["s_upper"])
    R_coord_upper=np.array(matrix_DIV_coord_upper["R_upper"])
    Z_coord_upper=np.array(matrix_DIV_coord_upper["Z_upper"])

    diff_R_lower=(R_coord_lower-R_probe)
    diff_R_upper=(R_coord_upper-R_probe)
    if Z_probe>0: #upper
	    diff_Z=(Z_coord_upper-Z_probe)
	    diff_total=(diff_R_upper**2+diff_Z**2)**0.5
	    index_min,=np.where(diff_total==min(diff_total))
	    s_probe=np.mean(s_coord_upper[index_min]) #take mean in the unlikely event that there is more than one index.
    else: #lower
	    diff_Z=(Z_coord_lower-Z_probe)
	    diff_total=(diff_R_lower**2+diff_Z**2)**0.5
	    index_min,=np.where(diff_total==min(diff_total))
	    s_probe=np.mean(s_coord_lower[index_min])
    return s_probe

if __name__ == '__main__':
    if True:
        # plot the following:
        plt.figure()
        # r_start_u,r_end_u,r_div_noshift_l
        plt.plot(r_div_noshift1,z_div_noshift1, color='k', alpha=0.2, label='r_div_noshift1', marker='o',
                 markersize=4)
        if False:
            plt.plot(r_start_u,z_start_u, color='blue', label='r_start_u', alpha=0.8, lw=3)
            plt.plot(r_end_u,z_end_u, color='orange', label='r_end_u', alpha=0.8, lw=3)
            plt.plot(r_div_noshift_l,z_div_noshift_l, color='green', label='r_div_noshift_l', alpha=0.8, lw=3)
            plt.plot(r_div_noshift, z_div_noshift, color='red', ls=':', label='r_div_noshift', alpha=0.8)

        if True:
            plt.plot(r_div_noshift1[0:15], z_div_noshift1[0:15], color='orange', ls='-.', label='r_div_noshift1[0:15]',
                     alpha=0.8)
            plt.plot(r_div_noshift[0:15],z_div_noshift[0:15], color='green', ls='--', label='r_div_noshift[0:15]', alpha=0.8)
        plt.legend()
    matrix_DIV_coord_lower_noshift,matrix_DIV_coord_upper_noshift=create_R_Z_s_coords(r_div_noshift,z_div_noshift)
    s_probe=locate_probe_s(0.540226997,1.515816087,matrix_DIV_coord_lower_noshift,matrix_DIV_coord_upper_noshift)
    print(s_probe)

    plt.figure()
    plt.plot(matrix_DIV_coord_lower_noshift['R_lower'], matrix_DIV_coord_lower_noshift['Z_lower'])
    plt.show()
    pass