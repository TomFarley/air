pro show_data,shot=shot,st=st,et=et,fi=fi,ll=ll,ul=ul,co=co,$
	xll=xll,xul=xul,no_loc=no_loc

if not keyword_set(fi) then fi=0.
if not keyword_set(shot) then shot=1
if not keyword_set(co) then if keyword_set(fi) then co=26 else co=27
if co lt 0 then co=0

exper='AUG'
path='./DATA/'
filter='*.*'
error=get_data( shot, exper, data, time, location, fn=fn, type=dat_typ,filter=filter,path=path)
if error ne 0 then error_msg,error,/halt
max_times=n_elements(time)
as = n_elements(location)

;t0=43.7
; time=(time-t0)*52.24/64+t0

;        data=smooth(float(data),2)

case dat_typ of
	0 : zt = 'raw data [digits]'
	1 : zt ='temperature [C]'
	2 : begin
			zt ='heat flux / MW/m**2'
			data=data*1.e-6
		end
	else : zt = 'a.u.'
end

if not keyword_set(st) then st=0.
if not keyword_set(et) then et=time(max_times-1)
	t_ind = where(time ge st and time le et,t_count)
	if t_count ge 2 then begin
		time=time(t_ind)
		data=data(*,t_ind)
	end

        ; time will be the x-axis further on
        data=rotate(data,4)

; average the data
	av = n_elements(time)/256

	if av gt 1 then begin
		to_show=fltarr(256,as)
		r_time=fltarr(256)
		print,'reduction by averaging: ', av
		for i= 0,255 do begin
			ind=Long(i*av)
			r_time(i)=total(time(ind:ind+av-1))/av
			for j = 0,as-1 do $
			to_show(i,j)=total(data(ind:ind+av-1,j))/av
			end
		end else begin
			r_time=time
			to_show=data
		end

if not keyword_set(xll) then xll = min(location)
if not keyword_set(xul) then xul = max(location)
loc_ind = where (location ge xll and location le xul,count)
if count ge 10 then begin
	to_show=to_show(*,loc_ind)
	location=location(loc_ind)
end

	profile=(to_show(0,*)+to_show(1,*)+to_show(2,*)+to_show(3,*))/4
	dummy=to_show(0,*)
	for i=1, n_elements(r_time)-1 do $
		dummy(*)=dummy+to_show(i,*)
	profile=dummy/n_elements(r_time)
 	init_out,co=co,fi=fi

        shot_str=strcompress(string(shot),/remove_all)
        pt=fn;+shot_str

	if keyword_set(no_loc) then location=indgen(n_elements(location))
	index=where (location ne 0)
	if not keyword_set(ll) then ll = 0
	if not keyword_set(ul) then ul = 0
	!P.charsize=2
		set_win,0,xs=450,ys=300
        show_3d,to_show(*,index),r_time,location(index),pt=pt,xt='time /s',$
                yt='divertor surface / mm',zt=zt,$
				ll=ll,ul=ul
		set_win,1,xs=450,ys=300
		surface,to_show(*,index),r_time,location(index),title=pt,xtitle='time /s',$
                ytitle='divertor surface / mm',ztitle=zt,$
				zrange=[ll,ul],az=45,zs=1
		set_win,2,xs=450,ys=300
		plot,location(index),to_show(0,index),yrange=[ll,ul],ys=1
		for i = 1, 10, 1 do oplot,location(index),to_show(i,index),line=i
;	if not fi then set_win,1
;	plot,location(o_ind),profile(o_ind)
	finish_o,fi=fi
; stop
end
