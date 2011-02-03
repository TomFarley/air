Short comments to THEODOR run and see - by Albrecht Herrmann
30.01.2003

unzip the file, keeping the directory structure (./DATA is needed)
and start idl in the directory of run_theo.pro

run load_knl (IDL>load_knl); DO this first to avoid idl error output
run run_theo (IDL>run_theo,/test)
use the switch test to see the input data and the result

then you should a selection window with the files in the data directory

the DATA folder containes example files to test theodor. 
calculating the heat flux from sim.tem you should get Gaussian spatial distribution withe a 2MW/m^2 peak load
	sim_ELM.tem has ELMs added with 50MW/m^2 and a duration of 1 ms, 10Hz. To get this results, aniso=0 (1D) 
	is required.

IDL programms to run theodor

theo_knl - compiled idl kernel
run_theo - prepares theo_knl, reads the data and the material parameters
load_knl - loads the compiled idl kernel
get_data - reads the (binary) data from file
error_msg - file with error messages
show_3d - graphic output as 3d color graphic
init_out - prepares the graphics output (window or file)
finish_o - finelizes the graphics output

sim_temp - calculates temporal temperature evolution from the 1D solution
		
wrt_temp - writes the sim_temp output to files
		ASCII file for ABAQus input (wrt_sim.asc)
		and as binary file for theodor input (wrt_sim.tem)


Parameters for simulation with/without ELMs
	q_0 = 2e6 	;W/m^2 ; maximum heat flux for simulation
	q_elm = 50e6	; W/m**2 elm heat flux
	n_time=5000.
	duration=1. ; s
	deltay = 0.005 ; m - distance along surface (pixel resolution)
	n_y = 36	;pixels along surface
	; ELM definitions
	sig=2		; width of the heat flux profile
	elm_freq=10 ; ELMs/s
	elm_dur=0.001; elm duration in s
Material parameters for simulation (use the same data in run_theo)
	lam=142			; W/m/K
	diff=62.e-6	    ; m^2/s
	heat_impact = SQRT(diff)/lam ; m**2/W * K/s**-0.5
	alpha_top = 144000.e10 ; W/m**2/K heat transmission at the front site
	NO temperature dependence !!!

Parameters (switches) to control theo_knl
; input parameter
; 	data : float temperature array (location,time)
;	time : float array (time)
;	location : float array (location)
;	d_target: parameter - target thickness in m
;	diff = float array (3) ; heat diffusion coefficient [0,500,1000 C];
;	lam = float array (3) ; heat conduction coefficient
;	alpha_bot = float parameter ; heat transmission coefficient at the bottom
;	alpha_top = float parameter ; heat transmission coefficient at the top
;
; parameter for the visiualisation
;	anim = 	long ; time index of the first bulk temperatur distribution be shown 
;			by the animate routine of idl.
;	max_frames = integer ; number of frames shown by the animation (default 150)
;			maximum number depends on the computer system and idl version
;	a_skip = 	integer  ; number of time steps to be skipe between animated T distributions (default=10)
;	x_Tb =	integer ; depth position for T(t) vector  (default: half depth; negative values -> 0)
;	y_Tb =  	integer ; lateral position for T(t) vector (default: half width; negative values -> 0)
;	ti_pr = 	long ; first time index to store T profiles in T_profiles (default: half time )
;	nr_pr=	long ; number of profiles stored in T_profiles (default: number of time steps / 100)
;	ll  =	float ; lower level for data visiualisation (default: auto scaling)
;	ul =	float ; upper level for data visiualisation (default: auto scaling)
;	co=	integer ; number of the idl color table (default: 26)
;
; key word parameter
;	test - switch to force output of parameters and graphics
;
; output parameter
;	T_profiles : Temperature profiles at y_Tb at time ....
; 	T_bulk : temporal evolution at the position x_Tb,y_Tb (x is depth, y parallel surface)



Data visialisation
The folder 
