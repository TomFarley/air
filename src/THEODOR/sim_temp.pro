; creates a temperature array according the 1 D solution
;	considering a heat transmission coefficient at the top


function sim_temp,data, time, location,heat_impact, alpha,$
				 elms=elms, noise=noise
; define the simulation parameters
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

; define the arrays
	data=findgen(n_y,n_time)
	location=Indgen(n_y)*deltay
	time=findgen(n_time)/(n_time-1)*duration
	auy = n_elements(data(*,0))
	n_time = n_elements(time)

; arteficial data (normalized temperature distribution)
	loc_dist=exp(-((auy/2-data(*,0))/sig/2)^2)
	for i=n_time-1,0,-1 do $
;		data(*,i) = sin(data(*,0)/(auy-1)*!Pi)*sqrt(time(i))
		data(*,i) = loc_dist*sqrt(time(i))
; arteficial noise
	if keyword_set(noise) then begin
		if noise ge 0.5 or noise le 0 then noise=0.1
		seed=1001L
		rand=randomu(seed,auy,n_time)
		data=data*(1+noise*(0.5-rand))
	end

; calculate the temperature distribution
; delatT = q * 2/sqrt(pi lam rho c) * sqrt(time)
;
	T_factor = q_0 * 2 / SQRT(!Pi) * heat_impact
	;print,'max data ',max(data)
	data= data * T_factor
; consider alpha
	for i = 0, n_time -1 do $
		data(*,i)=data(*,i)+loc_dist*q_0/alpha

if keyword_set(ELMs) then begin
; ELM simulation
; define 1 ELM - elm_array
	print,'adding ELMs'
		dt=time(1)-time(0)
		elm_inc=fix(1./elm_freq/dt+0.5) ; increment in time steps
		n_du = fix(elm_dur/dt+0.5)
;		n_de = 2000*n_du
		n_de=n_time
		if n_du le 0 then print,'time step lower than ELM duration'

		elm_dur=indgen(n_du) ; in time steps
		elm_decay=indgen(n_de)+2 ; in time steps

		elm_start=T_factor*q_elm/q_0*sqrt(elm_dur)
		elm_decay=max(elm_start)/sqrt(elm_decay)
		offset=Findgen(n_de)/(n_de-1)*elm_decay(n_de-1)
;		elm_decay=elm_decay-offset

		elm_arr=fltarr(auy,n_du+n_de)
		for i=0,n_du-1 do $
			elm_arr(*,i)=loc_dist*elm_start(i)
		for i=0,n_de-1 do $
			elm_arr(*,n_du+i)=loc_dist*elm_decay(i)

		elm_arr=elm_arr*sqrt(dt)
; ELM with alpha
;		elm_arr=findgen(auy,n_du)
		for i = 0, n_du-1 do $
			 elm_arr(*,i) = elm_arr(*,i) + loc_dist*q_elm/alpha


; add ELMS onto the time evolution
;
;		elm_per=n_elements(elm_arr(0,*))
		
		for i = 202, n_time-800, elm_inc do begin 
			data(*,i:n_time-1)=data(*,i:n_time-1)+elm_arr(*,0:n_time-i-1)
;			data(*,i:i+elm_per-1)=data(*,i:i+elm_per-1)+elm_arr(*,*)
		end
end
; shift the start of the heat puls by 50 steps

move_step=fix(n_time*0.05)
if move_step le 1 then move_step=2
data(*,move_step:n_time-1)=data(*,0:n_time-move_step-1)
data(*,0:move_step-1)=0.

return,0
end