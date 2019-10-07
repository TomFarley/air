;**************************************************************
;
; This function calculates a temperature evolution according the 1D solution
; for heat condution in a semi-infinite target.
;
; Heat transmission at the top can be added.
;
; ELMs are considered as heat pulses of duration elm_dur.
;
; Heating phase with q:
; T=2/sqrt(Pi) * 1./(sqrt(lam*rho*cp)) * q * sqrt(t)
;
; decay at t > t0
; T=2/sqrt(Pi) * 1./(sqrt(lam*rho*cp)) * q * (sqtr(t) - sqrt(t-t0))
;
; The resulting temperature evolution is teh sum of 1 D solution for
; contineous heating and heat pulses of duration elm_dur
;
; Arteficial noise can be added
;
; Albrecht Herrmann
; Max-Planck-Institut für Plasmaphysik
; Boltzmannstr. 2
; 85478 Garching
; Germany
; e-mail: Albrecht.Herrmann@ipp.mpg.de
; last modification: Januar 2005
;
;**************************************************************************
;

function get_temp_sim, data, time, location, heat_impact, alpha,$
						q_0,q_elm,elm_freq,elm_dur, sig,$
				 		elms=elms, decay=decay, noise=noise

	if NOT keyword_set(decay) then decay=0 else decay=1

	auy 	= n_elements(data(*,0))
	n_time 	= n_elements(time)

; arteficial data (normalized temperature distribution)
	loc_dist=exp(-((auy/2-data(*,0))/sig/2)^2)  ; Gaussian heat load distribution

	if decay ne 0 then puls_ind=fix(0.75*n_time) else puls_ind=n_time
	for i=puls_ind-1,0,-1 do $
		data(*,i) = loc_dist*sqrt(time(i))
	for i=puls_ind,n_time-1 do $
		data(*,i) = loc_dist*(sqrt(time(i)) - sqrt(time(i)-time(puls_ind-1)))

; arteficial noise
	if keyword_set(noise) then begin
		if noise ge 0.5 or noise le 0 then noise=0.1
		seed=1001L
		rand=randomu(seed,auy,n_time)
		data=data*(1+noise*(0.5-rand))
	end

; calculate the experimental parameters
; delatT = q * 2/sqrt(pi lam rho c)
;
	T_factor = q_0 * 2 / SQRT(!Pi) * heat_impact
	;print,'max data ',max(data)
	data = data * T_factor

; consider a top layer with DT=q/alpha
	for i = 0, puls_ind -1 do $
		data(*,i)=data(*,i)+loc_dist*q_0/alpha


if keyword_set(ELMs) then begin
; ELM simulation
; define 1 ELM - elm_array
	print,'adding ELMs'
		dt=time(1)-time(0)
		elm_inc=fix(1./elm_freq/dt+0.5)  ; increment in time steps
		n_du = fix(elm_dur/dt+0.5)	 ; number of ELM sliced for heating
;		n_de = 2000*n_du		 ; total number including cooling
		n_de=n_time
		if n_du le 0 then print,'time step lower than ELM duration'

		elm_dur=indgen(n_du) 		 	; in time steps
		elm_start=T_factor*q_elm/q_0*sqrt(elm_dur)  ; the heating phase

		elm_decay=indgen(n_de)+n_du 	; in time steps

		elm_decay=T_factor*q_elm/q_0*(sqrt(elm_decay) - sqrt(elm_decay-(n_du-1)))


;		offset=Findgen(n_de)/(n_de-1)*elm_decay(n_de-1)
;		elm_decay=elm_decay-offset

		elm_arr=fltarr(auy,n_du+n_de)
		for i=0,n_du-1 do $
			elm_arr(*,i)=loc_dist*elm_start(i)
		for i=0,n_de-1 do $
			elm_arr(*,n_du+i)=loc_dist*elm_decay(i)


; transform to real time
		elm_arr=elm_arr*sqrt(dt)


; ELM with alpha
;		elm_arr=findgen(auy,n_du)
		for i = 0, n_du-1 do $
			 elm_arr(*,i) = elm_arr(*,i) + loc_dist*q_elm/alpha


; add ELMS onto the top of the stationary time evolution
;

		elm_st=n_time*0.1
		elm_et=puls_ind*0.9
		for i = elm_st, elm_et, elm_inc do $
			data(*,i:n_time-1)=data(*,i:n_time-1)+elm_arr(*,0:n_time-i-1)

end

; shift the start of the heat puls by 50 steps

move_step=fix(n_time*0.05)
if move_step le 1 then move_step=2
data(*,move_step:n_time-1)=data(*,0:n_time-move_step-1)
data(*,0:move_step-1)=0.

return,0
end