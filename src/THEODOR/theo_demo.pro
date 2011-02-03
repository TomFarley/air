; *******************************************************************************
;
; run_theo prepares the run of theodor
;
;
;
;***************************************************************
; Function qflux - is the IDL version of the THEODOR CODE
;
;(THermal Energy Onto DivetOR)
;
; based on the Fortran version of Klaus Guenther (1992)
; written to evaluate the heat flux from surface temperature
; measurements at ASDEX Upgrade
;
; 2D code with temperature dependent material parameters
; The calculation is done in heat potential terms
;
; assumtions:
; The target is assumed to be uniformly thick.
; The time step can be changed during the measurement.
; The front and backside edge condition is heat transmission.
;
; written by
; Albrecht Herrmann
; Max-Planck-Institut fuer Plasmaphysik
; Boltzmannstr. 2
; 85478 Garching
; Germany
; e-mail: Albrecht.Herrmann@ipp.mpg.de
;
;
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
;	x_Tb =	    float; depth position for T(t) vector  (default: half depth; negative values -> 0)
;	y_Tb =  	float ; lateral position for T(t) vector (default: half width; negative values -> 0)
;	ti_pr = 	long ; first time index to store T profiles in T_profiles (default: half time )
;	nr_pr=	long ; number of profiles stored in T_profiles (default: number of time steps / 100)
;	ll  =	float ; lower level for data visiualisation (default: auto scaling)
;	ul =	float ; upper level for data visiualisation (default: auto scaling)
;	co=	integer ; number of the idl color table (default: 26)
;
; key word parameter
;	test - switch to force output of parameters and graphics
;	foll - follows the calculation steps (output every 500 steps)
;	show - displays input data and results (independent on test)
;
; output parameter
;	T_profiles : Temperature profiles at y_Tb at time ....
; 	T_bulk : temporal evolution at the position x_Tb,y_Tb (x is depth, y parallel surface)
;
;
; theo_demo
;
; runs a demonstration for theo with simulated temperature files
;
;
;
;**************************************************************



function get_params, diff,lam,alpha_top,alpha_bot,d_target,heat_impact,aniso

; define the material parameters
; CFC
;	diff=[216.,42.,26.] ; mm^2/s
;	diff=[246.,62.,35.] ; mm^2/s	; Valerias JET data
	diff=[62.,62.,62.] ; mm^2/s  ; only for simulation

	diff=diff*1.e-6	    ; m^2/s
;	lam=[240.,112.,80.] ; W/m/K
;	lam=[318.,183.,122.] ; W/m/K	; Valerias JET data
	lam=[142.,142.,142.] ; W/m/K	; only for simulation
;	temp=[0,500,1000] ; C

; EK98
;	diff=[80.,21.,15.] ; mm^2/s
;	diff=diff*1.e-6	   ; m^2/s
;	lam=[115.,63.,49.] ; W/m/K
;	temp=[0,500,1000] ; C

	alpha_top = 144000.e10 ; W/m**2/K heat transmission at the front site
	alpha_bot = 100.  ; W/m**2/K heat transmission at the back site
; define the target thickness
	d_target=0.02 ;m
	heat_impact = SQRT(diff(1))/lam(1) ; m**2/W * K/s**-0.5
; anisotropy D_y/D_x
	aniso=0.0

return,0
end


;**************************************************************

pro theo_demo
;**************************************************************



;************  simulate temperature data *********************
print,'calculating test data ...'
wrt_simul,/decay,/elm

test=1
foll=1
sm=0


;************  read material parameters *********************
; should be the same as in wrt_simul.pro
error = get_params(diff,lam,alpha_top,alpha_bot,d_target,heat_impact,aniso)



;************ get the first simulation file (equidistant time step)
fn='wrt_sim.tem'
shot=1l
r_file,fn,data,location,time,dat_typ=type,error=error

;************ prepare the monitor parameters
;monitor the temperature inside the bulk at position x_tb and y_tb (in depth units)
;for all time points - output: t_bulk(time)
x_tb = d_target*0.1
y_tb=location(n_elements(location)/2)

;monitor temperature profiles inside the targets at position y_tb and time ti_pr
; ti_pr is a vector with N timevalues. For each time a profile is written to:
; output:t_profiles(N,100)
;ti_pr=rebin(time,long(n_elements(time)/1000),/sample)

; prepare the animation with xinteranimate
; anim=1000	; first frame for animation
; max_frames = 50 ; default - 150
; ll = 270 ; lower level for visiualisation (default - auto scaling)
; ul = 500   ; upper level for visiualisation
; a_skip = 1 ; default - equaly distributed animation (a_skip=fix(n_elements(time)/max_frames))

print,'calculating equidistant data'
; calculate the heat flux
		qflux=theo_mul( data, time, location, d_target, $
			alpha_bot, alpha_top, diff, lam, aniso, $
			test=test, foll=foll, h_pot=h_pot,$
			anim=anim,a_skip=a_skip,max_frames=max_frames,$
			T_bulk=T_bulk,x_Tb=x_Tb,y_Tb=y_Tb,$
			t_profiles=t_profiles, ti_pr=ti_pr,$
			ll=ll,ul=ul,co=co)

;write data to file

dot_pos=rstrpos(fn,'.')
if dot_pos gt 0 then fn=strmid(fn,0,dot_pos)
	fn=fn+'.hef'
	type=2
	print,'writing data to file: ',fn
      		close,1
		as=long(n_elements(location))
		max_times=long(n_elements(time))
        	openw,1,fn
        	writeu,1,long(shot),as,max_times,type
	       	writeu,1,qflux,time,location
	       	writeu,1,float(diff),float(lam),float(mean(alpha_top)),$
	       				float(alpha_bot),float(d_target),$
	       				float(heat_impact),float(aniso),float(sm)
        close,1


; get the multi time step simulation file (equidistant time step
fn='wrt_mul.tem'
r_file,fn,data_mul,location,time_mul,dat_typ=type,error=error

print,'calculating multi time step data'
; calculate the heat flux
		qflux_mul=theo_mul( data_mul, time_mul, location, d_target, $
			alpha_bot, alpha_top, diff, lam, aniso, $
			test=test, foll=foll, h_pot=h_pot,$
			anim=anim,a_skip=a_skip,max_frames=max_frames,$
			T_bulk=T_bulk,x_Tb=x_Tb,y_Tb=y_Tb,$
			t_profiles=t_profiles, ti_pr=ti_pr, $
			ll=ll,ul=ul,co=co)

;write data to file

dot_pos=rstrpos(fn,'.')
if dot_pos gt 0 then fn=strmid(fn,0,dot_pos)
	fn=fn+'.hef'
	type=2
	print,'writing data to file: ',fn
      		close,1
		as=long(n_elements(location))
		max_times=long(n_elements(time_mul))
        	openw,1,fn
        	writeu,1,long(shot),as,max_times,type
	       	writeu,1,qflux_mul,time_mul,location
	       	writeu,1,float(diff),float(lam),float(mean(alpha_top)),$
	       				float(alpha_bot),float(d_target),$
	       				float(heat_impact),float(aniso),float(sm)
        close,1
end
