pro get_ir,sh,time,data,t,d,integtime,cphot,framerate,rawd,lambda,rawnuc=rawnuc, run_check=run_check

;subtracts the frame defined by time given in varible t to perform NUC
;converts the photon counts to temperature using the calibaration (raw2temp)

;NUC can be performed using several frame before the start of the shot
;or by using the first frame only
if(n_elements(t) eq 2) then begin
        ;if 'nuc' time range given then average frames over window
	print,' For nuc subtaction, average over time',t
	av_image,time,data,t,d,framerate
endif else begin
        ;find frame closest to desired 'nuc' time
	q=where(abs(time-t) eq min(abs(time-t)))
	t=time(q(0))
 	d=data(*,*,q(0))
endelse

rawd = d ;raw data array

;don't do this part if checking the alignment with run_check
if not keyword_set(run_check) then begin
ftype=size(sh,/type)
if(ftype ne 7) then begin 
        if (keyword_set(rawnuc)) then d = d
        d = d> 0 ;make them zero if they are negative
	;convert the counts to temp using the calibration (see raw2temp_aug.pro)
	;print, integtime
        d=raw2temp_aug(sh,d,integtime) ;returns temp profile to variable d      
        d=float(d)
endif else begin
	d=float(d)
endelse
endif

end
