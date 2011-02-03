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
; The time step is not changed during the measurement.
; The front and backside edge condition is heat transmission.
;
; written by
; Albrecht Herrmann
; Max-Planck-Institut f’r Plasmaphysik
; Boltzmannstr. 2
; 85478 Garching
; Germany
; e-mail: Albrecht.Herrmann@ipp.mpg.de
;
; Januar 2001; March 2002
;
; 14.12.2002
; Animation is included and a bug at the edges fixed
;
; Juni 2004
; alpha_top can be an array with the same dimension as location (if not alpha_top is used as scalar)
;
; December 2005
; Factor checked (y smaller than x ?)
;
; November 2007, multi time step improved (Congrid)
;
; The input array structure should be arr(y,time). If not it is rotated (and back at the end)
;
;**************************************************************
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
;	x_Tb =	integer ; depth position for T(t) vector  (default: half depth; negative values -> 0)
;	y_Tb =  	integer ; lateral position for T(t) vector (default: half width; negative values -> 0)
;	ti_pr = 	long ; first time index to store T profiles in T_profiles (default: half time )
;	ll  =	float ; lower level for data visiualisation (default: auto scaling)
;	ul =	float ; upper level for data visiualisation (default: auto scaling)
;	co=	integer ; number of the idl color table (default: 26)
;
; key word parameter
;	test - switch to force output of parameters and graphics
;	foll - follows the calculation steps (output every 500 steps)
;	show - displays input data and results (independent on test)
;   lun  - Logical unit number for message output, default lun=-1
;
; output parameter
;	T_profiles : Temperature profiles at y_Tb at time ....
; 	T_bulk : temporal evolution at the position x_Tb,y_Tb (x is depth, y parallel surface)
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

	alpha_top = 144000;.e10 ; W/m**2/K heat transmission at the front site
	alpha_bot = 100.  ; W/m**2/K heat transmission at the back site
; define the target thickness
	d_target=0.02 ;m
	heat_impact = SQRT(diff(1))/lam(1) ; m**2/W * K/s**-0.5
; anisotropy D_y/D_x
	aniso=0.5

return,0
end


;**************************************************************

pro run_theo, shot, exper, test=test, foll=foll, show=show, sm=sm,fn=fn,$
		anim=anim,a_skip=a_skip,max_frames=max_frames,$
		T_bulk=T_bulk,x_Tb=x_Tb,y_Tb=y_Tb,$
		t_profiles=t_profiles, ti_pr=ti_pr,$
		ll=ll,ul=ul,co=co

;**************************************************************



; get temperature data and material parameters
if n_elements(shot) eq 0 then shot='1' else shot=strcompress(string(shot),/remove_all)

case chk_sys() of
     'PC': path='D:\IDL_alh\'
     'UNIX': path='/u/alh/'
     ELSE : path=''
endcase

filter='*.tem'
error = get_data(shot,exper,data,time,location,fn=fn,type=type,path=path,filter=filter)
if error ne 0 then err_msg,error,/halt
if type ne 1 then err_msg,6,/halt


if keyword_set(show) then begin
	    	set_win,1
		!P.multi=0
		shade_surf,data,az=45,title='surface temperature'
end

error = get_params(diff,lam,alpha_top,alpha_bot,d_target,heat_impact,aniso)

print,'got the data'


; calculate the heat flux
qflux=theo_mul( data, time, location, d_target, $
		alpha_bot, alpha_top, diff, lam, aniso, $
		test=test, foll=foll, h_pot=h_pot,$
		anim=anim,a_skip=a_skip,max_frames=max_frames,$
		T_bulk=T_bulk,x_Tb=x_Tb,y_Tb=y_Tb,$
		t_profiles=t_profiles, ti_pr=ti_pr, $
		ll=ll,ul=ul,co=co)

if keyword_set(show) then begin
	auy = n_elements(data(*,0))
	n_time = n_elements(time)
	set_win,1
		!P.multi=[0,1,2,0,0]
		max_v=max(data(*,n_time-1),max_ind)
		print,max_v,max_ind
		max_v=max(qflux(*,n_time-1),max_ind)
		print,max_v,max_ind
		max_ind=20
		plot,time,qflux(max_ind,*);,xrange=[0,0.05]
		oplot,time,qflux(19,*),line=1
	set_win,2
		!P.multi=[0,1,2,0,0]
		shade_surf,hpot2temp(h_pot),title='end temperature',az=45
		shade_surf,qflux,title='heat flux',az=45
	set_win,3
		!P.multi=[0,1,2,0,0]
		shade_surf,diffstar(h_pot),title='diffstar'
		plot,qflux(*,n_elements(time)-1)
		oplot,qflux(*,100),line=1
		oplot,qflux(*,1),line=2
end

;write data to file

sm=0
dot_pos=rstrpos(fn,'.')
if dot_pos gt 0 then fn=strmid(fn,0,dot_pos)
	fn=fn+'.hef'
	type=2
	print,'writing data to file: ',fn
      		close,1
		as=long(n_elements(location))
		max_times=long(n_elements(time))
        	openw,1,fn
        	writeu,1,shot,as,max_times,type
	       	writeu,1,qflux,time,location
	       	writeu,1,float(diff),float(lam),float(mean(alpha_top)),$
	       				float(alpha_bot),float(d_target),$
	       				float(heat_impact),float(aniso),float(sm)
        close,1
end

