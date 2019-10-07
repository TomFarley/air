pro r_file,fn,data,ort,time,dat_typ=dat_typ,shot=shot,error=error

		error=0
        print,'reading data from file ',fn
        as=1l
        max_times=1l
        shot=0l
	dat_typ=0
        close,1
        openr,1,fn
        readu,1,shot,as,max_times,dat_typ
	if dat_typ ge 5 then begin
			byteorder,dat_typ
			if dat_typ le 4 then begin
				reorder = 1
				print,'the bytes will be swaped'
;				byteorder,typ
				byteorder,as,/lswap
				byteorder,max_times,/lswap
				byteorder,shot,/lswap
			end else begin
				print,'the selected file stores NO temperature data !'
				close,1
				error=5
;				goto,exit
			end
		end else reorder=0
	if as ge 1024 then begin
	    ay = fix(as/1024)
	    as = as - 1024l*ay
	    data=fltarr(as,ay,max_times)
	    print,'the file stores a 3D array (x,y,time): ',as,ay,max_times
;	    error=1
	end else begin
	    data=fltarr(as,max_times)
	    print,'the file stores a 2D array (x,time)',as,max_times
	end
        time=fltarr(max_times)
	;if dat_typ eq 0 then data=intarr(as,max_times) else

        ort=fltarr(as)
        readu,1,data,time,ort
	if reorder eq 1 then begin
		byteorder,data,/ftoxdr
		byteorder,ort,/ftoxdr
		byteorder,time,/ftoxdr
	end
       if dat_typ eq 2 then begin
            diff=findgen(3)
	    lam=diff
	    alpha_top=1.
	    alpha_bot=1.
	    d_target=1.
            heat_impact=1.
	    aniso=1.
	    sm=1.
            readu,1,diff,lam,alpha_top,$
            alpha_bot,d_target,$
            heat_impact,aniso,sm
            close,1
	    if reorder eq 1 then begin
			byteorder,diff,/ftoxdr
			byteorder,lam,/ftoxdr
			byteorder,alpha_top,/ftoxdr
			byteorder,alpha_bot,/ftoxdr
			byteorder,d_target,/ftoxdr
			byteorder,heat_impact,/ftoxdr
			byteorder,aniso,/ftoxdr
			byteorder,sm,/ftoxdr
	    end
	    print,'diff: ',diff
	    print,'lam: ',lam
	    print,'alpha_top: ',alpha_top
	    print,'alpha_bot: ',alpha_bot
	    print,'d_target: ',d_target
            print,'heat_impact: ',heat_impact
	    print,'aniso: ',aniso
	    print,'sm: ',sm
	end else close,1
exit:
end
