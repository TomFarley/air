; writes temperature data to be used by ABAQUS calculation

pro wrt_sim
	skip = 40	; to reduce the array
	lam=142			; W/m/K
	diff=62.e-6	    ; m^2/s
	heat_impact = SQRT(diff)/lam ; m**2/W * K/s**-0.5
	alpha_top = 144000.e10 ; W/m**2/K heat transmission at the front site
; calculate the temperature array
	error = sim_temp(data, time, location, heat_impact, alpha_top,/elms)

;show the array
	if !D.name eq 'WIN' then window,4
	shade_surf,data,az=45,title='surface temperature'

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
	print,'wrote to file: ',fn

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


end
