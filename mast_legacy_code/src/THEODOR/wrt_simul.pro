
pro wrt_simul,elm=elm,decay=decay,$
		alpha_top=alpha_top,lam=lam,diff=diff



if not keyword_set(elm) then elm = 0 else elm = 1
if not keyword_set(decay) then decay = 0 else decay = 1
; material data (constsnt)
if not keyword_set(lam) then lam = 142			; W/m/K
if not keyword_set(diff) then diff = 62.e-6	    ; m^2/s
	heat_impact = SQRT(diff)/lam ; m**2/W * K/s**-0.5
if not keyword_set(alpha_top) then alpha_top = 144000.e10 ; W/m**2/K heat transmission at the front site

; time
	n_time   = 10000.	; number of time points
	duration = 1.5 		; s

; define the simulation parameters
	deltay 	 = 0.005 	; 	m - distance along surface (pixel resolution)
	n_y = 36			;	pixels along surface
	sig=2				; 	width of the gaussian heat flux profile
	q_0      = 2e6 		;	W/m^2 ; maximum heat flux (in between ELMs)

; ELM definitions
	q_elm    	= 	50e6	; W/m**2 elm heat flux
	elm_freq	=	10 		; ELMs/s
	elm_dur 	=	0.001	; elm duration in s


; define the arrays
	data	 =	findgen(n_y,n_time)
	location =	Indgen(n_y)*deltay
	time	 =	indgen(n_time)/(n_time-1)*duration

; to calculate the array with multi time step
	skip = 10	; to reduce the array
	st	 = 0.4		; start time for full time resolution
	et	 = 1.3		; end time for full time resolution

; calculate the temperature array
	error = get_temp_sim(data, time, location, heat_impact, alpha_top,$
					q_0,q_elm,elm_freq,elm_dur,sig,elm=elm,decay=decay)

;show the array
;	if !D.name eq 'WIN' then window,4
;	shade_surf,data,az=45,title='surface temperature'

; write the data to ASCII file
	n_time=n_elements(time)
	fn='wrt_sim.asc'
	close,1
	openw,1,fn
	for i=0,n_time-1 do $
		printf,1,format='(F8.4,36F8.1)',time(i),data(*,i)
	close,1
	print,'wrote to file: ',fn

; write the file to alh_jet data structure
	fn='wrt_sim.tem'
	print,'writing data to file: ',fn
      	close,1
		as=long(n_elements(location))
		max_times=long(n_elements(time))
			type=1
			shot=1l
        	openw,1,fn
        	writeu,1,shot,as,max_times,type
	       	writeu,1,data,time,location
	       	writeu,1,diff,lam,alpha_top
        close,1


; and now write reduced arrays

; 1st. skipping
	fn='wrt_skip.tem'
	new_l=long(max_times/skip)
	new_ind=indgen(new_l)*skip
	new_data=data(*,new_ind)
	new_time=time(new_ind)
       	openw,1,fn
        	writeu,1,shot,as,new_l,type
	       	writeu,1,new_data,new_time,location
	       	writeu,1,diff,lam,alpha_top
        close,1
	print,'wrote reduced data to file: ',fn

; 2nd averaging
	fn='wrt_av.tem'
	for i = 1 , skip -1 do new_data=new_data+data(*,new_ind+i)
	new_data=new_data/skip
     	openw,1,fn
        	writeu,1,shot,as,new_l,type
	       	writeu,1,new_data,new_time,location
	       	writeu,1,diff,lam,alpha_top
        close,1
	print,'wrote to file: ',fn


; 3d. multi time step
	fn='wrt_mul.tem'
; calculate the indices
	as =n_elements(data(*,0))
	max_times = n_elements(time)
	st_ind    = where (time lt st,stc)
	if stc ge 10*skip then begin
		stc=long(stc/skip)
		time(stc*(skip-1):stc*skip-1)=rebin(time(0:stc*skip-1),stc,/sample)
		time=time(stc*(skip-1):max_times-1)
		data(*,stc*(skip-1):stc*skip-1)=rebin(data(*,0:stc*skip-1),as,stc,/sample)
		data=data(*,stc*(skip-1):max_times-1)
	end
	et_ind = where (time ge et,etc)
	if etc ge 10*skip then begin
		etc=long(etc/skip)
		time(et_ind(0):et_ind(0)+etc-1)=rebin(time(et_ind(0):et_ind(etc*skip-1)),etc,/sample)
		time=time(0:et_ind(0)+etc-1)
		data(*,et_ind(0):et_ind(0)+etc-1)=rebin(data(*,et_ind(0):et_ind(etc*skip-1)),as,etc,/sample)
		data=data(*,0:et_ind(0)+etc-1)
	end
	max_times=n_elements(time)
    openw,1,fn
        writeu,1,shot,as,max_times,type
	    writeu,1,data,time,location
	    writeu,1,diff,lam,alpha_top
    close,1
	print,'wrote multi time step to file: ',fn
end
