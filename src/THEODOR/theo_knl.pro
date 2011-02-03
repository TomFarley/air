;***************************************************************
;
; IDL version of the THEODOR (THermal Energy Onto DivetOR) CODE
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
;
;
; written by
; Albrecht Herrmann
; Max-Planck-Institut für Plasmaphysik
; Boltzmannstr. 2
; 85478 Garching
; Germany
; e-mail: Albrecht.Herrmann@ipp.mpg.de
;
; Januar 2001; March 2002 
;
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
; key word parameter
;	test - switch to force output of parameters and graphics
;
; output parameter
;	qflux : float heat flux array (location, time)

;**************************************************************


function heatpotential,t_data
common params, cc, offset, hts, ad, bd, tratio, tc0, td0, plusac
	t_data=(t_data+offset)/tc0
	h_pot=t_data*(CC + 0.5 * T_data)/(1+T_data)
return,h_pot
end


function hpot2temp,h_pot
common params, cc, offset, hts, ad, bd, tratio, tc0, td0, plusac
	hpc=h_pot-cc
	temp=Tc0*(hpc+SQRT(hpc*hpc+2*h_pot))
return, temp
end

function diffstar,h_pot
common params, cc, offset, hts, ad, bd, tratio, tc0, td0, plusac
	hpc=h_pot-cc
	ww= SQRT(hpc*hpc+2.*h_pot)
	if not plusac then ww=-ww
	d_star=1.+tratio*(hpc+ww)
	d_star=AD+BD/(d_star*d_star)
return, d_star
end

function weight,h_pot
common params, cc, offset, hts, ad, bd, tratio, tc0, td0, plusac
	hpc=h_pot-cc
	ww= SQRT((hpc*hpc)+2.*h_pot)
	if not plusac then ww=-ww
	weight=ww/(ww+ww+ww+hts*(ww+hpc+1.))

return,weight
end

function klg_fit,data,a,b,T0,t_max
	data=float(data)
	y1=data(0)
	y2=data(1)
	y3=data(2)
;initialize the variables
	a=0
	b=0
	t0=0
	t_max=0

; check limits (errors)
; increasing ?
if y3 gt y2 and y1 gt y2 then return,1
;constant and not negative?
if y1 eq y2 and y2 eq y3 then begin
			B=0
			T0=500.
			if y1 le 0 then return,2 else a=y1
			return,0
	end
;convex
	if  2*y2 ge y1+y3 then return,3

; start the fit
	W=SQRT((y2-y3)/(y1-y3))
	phi=acos(W)
	chi=cos((!Pi-phi)/3)/W
	T0=500./(chi-1)
	b=chi*chi/(chi*chi-1)*(y1-y2)
	a=y1-b
	if a lt 0 then begin
		t_max=T0*(SQRT(-B/A)-1.)
		return, 4
	end
	if a eq 0 then a=0.001 * y3
	return,0
end



;***********************************************************************

function theo_knl, data, time, location, d_target, $
		alpha_bot, alpha_top, diff, lam, aniso, $
		test=test, t_profiles=t_profiles, h_pot=h_pot

;***********************************************************************

common params, cc, offset, hts, ad, bd, tratio, tc0, td0, plusac

print,'2D heat flux calculation, A.Herrmann, IPP Garching; March 2002'
;print,'*****************************************************'
;print,' 2D heat flux calculation with (FTCS method)'
;print,' temperature dependend material coefficients'
;print,' written by K.Guenter, A. Herrmann, IPP Garching'
;print,' IDL version 2.0; march 2002 '
;print,'*****************************************************'


; parameters of calculation

	If NOT keyword_set(T_profiles) then get_profs=0 else get_profs=1

;define the critical Diffusivity
	dstar=0.35
	;dstar=0.25

;maximum number of meshes in depth
	max_meshes = 150
;	multiply = [6,10] ; not implemented yet

; minimum number of meshes required in dpth
	min_meshes = 5

; define the anisotropy D_x/D_y
	;aniso=0.5 ; is an input parameter

; normalized heat transmission
	htrans_b	= alpha_bot/lam(0) ; 1/m
	htrans_f	= alpha_top/lam(0)

; define the output array
	auy = n_elements(data(*,0))
	n_time = n_elements(time)
	qflux=fltarr(auy,n_time)
	as = n_elements(location)
	if as ne auy then print,'missmatch between location length and data size'

; averaged surface mesh width
	deltay = total(location(1:as-1)-location(0:as-2))/(as-1)

;fit parameters for diff and lam calculation
;	Y(T)=a+b/(1+T/T0)**2
; 	[lambda] = W/m/K	[Diff] = m**2/s
	error=klg_fit(diff,Ad0,Bd0,Td0,t_maxd)
	if keyword_set(test) then begin
		print,'fit data for diffusity'
		print,'a, b, T0, T_max: ',ad0, bd0,td0,t_maxd
	end
	if error gt 0 then err_msg,error
	if error ne 0 and error ne 4 then stop

	error=klg_fit(lam,Ac0,Bc0,Tc0,T_maxc)
	if keyword_set(test) then begin
		print,'fit data for conductivity'
		print,'a, b, T0, T_max: ',ac0, bc0,tc0,t_maxc
	end
	if error gt 0 then err_msg,error
	if error ne 0 and error ne 4 then stop


	if ac0 ge 0 then plusac=1 else plusac=0	; upper temperature limit if ac < 0
	if ad0 ge 0 then plusad=1 else plusad=0
	cflux0=2*ac0*tc0
	tratio=tc0/Td0
	CC=1.+BC0/AC0
	htrans_b=cc*htrans_b
	htrans_f=cc*htrans_f
	cc=cc/2.

; define arrays for monitoring
	T_bulk=time
	T_bulk(*)=0.

	T_unit = 'C'
	if T_unit eq 'C' then offset = 0 else offset=-273

; get the time step and optimum mesh width
	dt=time(1)-time(0)

;tile dependent parameter
new_dstar:

	dx=sqrt(diff(0)*dt/dstar) ; optimum mesh width
	aux=fix(d_target/dx+0.5) ; mesh number

	if aux gt max_meshes then begin		; adjust the mesh numbers
		dstar = dstar * (max_meshes*dx/d_target)^2
		print,'dstar changed to: ',dstar
		goto,new_dstar
	end

	if aux le min_meshes then err_msg,8,/halt

	relwidth=1./aux
	delta=d_target*relwidth	; elementary mesh width in x direction
	star=delta^2
	factor=aniso*star/deltay^2
	star=dt/star

	if keyword_set(test) then begin
		print,'mesh number ',aux
		print,'relwidth ',relwidth
		print,'elementary mesh width ',delta
		print,'mesh width in y direction ',deltay
		print,'time steps: ',n_time
		print,'time increment: ',dt
	end

	ay1=auy-1
	ay2=auy-2
	ay3=auy-3
	ax1=aux-1
	ax2=aux-2
	ax3=aux-3

; calculate dimenionless parameters
	adstar=ad0*star
	bdstar=bd0*star
	cfstar=cflux0/(delta+delta)

; !!!!! changed by alh 13.01.2001 to make it consistent with modelling
; 	check the reason

	htstar_b=htrans_b*delta ; *2
	htstar_f=htrans_f*delta ; *2

;	Initializing the 2-D field of heat potential "h_pot" by assuming
;	that the initial x-dependence (depth) of heat potential can be described
;	by a quadratic parabola between top and bottom, where only for the latter
;	the heat potential is assumed to be constant laterally and is estimated
;	by the mean value for the surface of each tile, excluding the very edges

 	h_pot=fltarr(auy,aux)
	top=heatpotential(data(1:auy-2,0))
	hpc=total(top)/n_elements(top)	; averaged heat potential at the surface
	x=0.
	for j=0,ax1 do begin
		h_pot(1:ay2,j)=top+(hpc-top)*x^2
		x=x+relwidth
	end

; no losses at the edge
	h_pot(0,*)=h_pot(1,*)
	h_pot(ay1,*)=h_pot(ay2,*)

	hpcoolant=hpc	; coolant heat potential = averaged surface h_pot at t=0

; big time loop

	ad=adstar
	bd=bdstar
	f=factor
	fdouble=2*f
	cflux=cfstar
;	hts=htstar
	hpc=hpcoolant

if keyword_set(test) then begin
	print,'adstar ',ad
	print,'bdstar ',bd
	print,'factor ',f
	print,'cflux ',cflux
;	print,'htstar',hts
	print,'hpcoolant ',hpc
end

 Print,'starting calculations at: ',systime(0)
	i = 0 ; used for T monitoring
; *****************************************************************************


    for j = 1L, 100 do begin ;Long(n_elements(time)-1) do begin
	delta= (h_pot(1:ay2,0:ax3)$
			- 2*h_pot(1:ay2,1:ax2)$
			+ h_pot(1:ay2,2:ax1)$
			+ (H_pot(0:ay3,1:ax2)$
			- 2*h_pot(1:ay2,1:ax2)$
			+ h_pot(2:ay1,1:ax2)) * f)$
			*diffstar(h_pot(1:ay2,1:ax2))

; edges
	hpleft= (h_pot(0,0:ax3)$
			- 2*h_pot(0,1:ax2)$
			+ h_pot(0,2:ax1)$
			+ (H_pot(1,1:ax2)$
			- h_pot(0,1:ax2)) * 2*f) $
			*diffstar(h_pot(0,1:ax2))

	hprigth= (h_pot(ay1,0:ax3)$
			- 2*h_pot(ay1,1:ax2)$
			+ h_pot(ay1,2:ax1)$
			+ (H_pot(ay2,1:ax2)$
			- h_pot(ay1,1:ax2)) * 2*f) $
			*diffstar(h_pot(ay1,1:ax2))


; the heat potential for this time step
	h_pot(1:ay2,1:ax2)=h_pot(1:ay2,1:ax2)+delta     ; bulk
	h_pot(0,1:ax2) = h_pot(0,1:ax2) + hpleft	; left edge
	h_pot(ay1,1:ax2) = h_pot(ay1,1:ax2) + hprigth	; right edge

; next surface temperature to the top

	h_pot(*,0)=heatpotential(data(*,j))

; bottom corners (are handeled by the edge already
;	h_pot(0,ax2)=h_pot(1,ax2)
;	h_pot(ay1,ax2)=h_pot(ay2,ax2)

; bottom
	hts=htstar_b
	h_pot(0:ay1,ax1)=HPC + (4*h_pot(0:ay1,ax2)-H_POT(0:ay1,ax3)-3*hpc)$
		* weight((hpc+h_pot(0:ay1,ax1))/2)
; front
	hts=htstar_f
	t_aux=h_pot(1:ay2,0)
	h_pot(1:ay2,0)=t_aux + (4*h_pot(1:ay2,1)-h_pot(1:ay2,2)-3*t_aux)$
		* weight(h_pot(1:ay2,1)+(t_aux-h_pot(1:ay2,2))/2)

; top corners
	h_pot(0,0)=h_pot(1,0)-0.333333*(4*(h_pot(1,1)-h_pot(0,1))$
				- (h_pot(1,2)-h_pot(0,2)))
	h_pot(ay1,0)=h_pot(ay2,0)-0.333333*(4*(h_pot(ay2,1)-h_pot(ay1,1))$
				- (h_pot(ay2,2)-h_pot(ay1,2)))

; surface heat flux
 	qflux(1:ay2,j)= cflux*(3.*h_pot(1:ay2,0)-4*h_pot(1:ay2,1)+h_pot(1:ay2,2))

; monitore temperature depth profiles
	if get_profs then if time(j) ge ti_pr and i lt nr_pr then begin
			t_profiles(i,*)=h_pot(20,*)
			i=i+1
		end

; store the temperature at a fixed position for monitoring
	;T_bulk(j)=h_pot(20,0)

end; for big time step loop

 Print,'finished calculations at: ',systime(0)

;*******************************************************************************

	;T_bulk=hpot2temp(T_bulk)
	if get_profs then t_profiles=hpot2temp(t_profiles)

return,qflux

exit:

return,-1

end
