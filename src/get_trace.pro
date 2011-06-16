pro get_trace,dim,rawd,theo,temp,stemp,o,n,i,j,l,satpix,line=line,area=area,disp=disp,nline=nline, run_check=run_check

common data,d,width,height,xst,yst,comp,dnuc,pix,s

;initialise variables
satpix = 0

if(keyword_set(line) or keyword_set(area)) then begin
	if(not keyword_set(area)) then begin
		if(keyword_set(disp)) then get_line,theo,o,n,i,j,l,r,s,pix,$
				/disp else get_line,theo,o,n,i,j,l,r,s,pix
  	endif else begin
		;loop through theo lines theo((r,phi,z),(dr,dphi,dz),(start,end))
		if not keyword_set(nline) then nline=20
		rr=findgen(nline)/(nline-1)*(theo(0,0,1)-theo(0,0,0))+theo(0,0,0)
		drr=findgen(nline)/(nline-1)*(theo(0,1,1)-theo(0,1,0))+theo(0,1,0)
		phir=findgen(nline)/(nline-1)*(theo(1,0,1)-theo(1,0,0))+theo(1,0,0)
		dphir=findgen(nline)/(nline-1)*(theo(1,1,1)-theo(1,1,0))+theo(1,1,0)
		zr=findgen(nline)/(nline-1)*(theo(2,0,1)-theo(2,0,0))+theo(2,0,0)
		dzr=findgen(nline)/(nline-1)*(theo(2,1,1)-theo(2,1,0))+theo(2,1,0)

		theol=[[rr(0),phir(0),zr(0)],[drr(0),dphir(0),dzr(0)]]

		if(keyword_set(disp)) then get_line,theol,o,n,i,j,l,r,sl,pixl,/disp $
					else get_line,theol,o,n,i,j,l,r,sl,pixl
    		pixall=pixl
		sall=sl
		
		if (keyword_set(run_check) eq 1 and sall[0] eq 0) then begin
			xyouts, 0.05, 0.05, 'FAIL', /normal, charsize=1.75, color=truecolor('black')
		endif

		for it=1,n_elements(rr)-1 do begin
			theol=[[rr(it),phir(it),zr(it)],[drr(it),dphir(it),dzr(it)]]
			if(keyword_set(disp)) then get_line,theol,o,n,i,j,l,r,sl, $ 
				pixl,/disp else get_line,theol,o,n,i,j,l,r,sl,pixl

			if(n_elements(pixl) gt 2) then begin
				if not keyword_set(run_check) then begin
					pixall=[pixall,pixl]
					sall=[sall,sl]
				endif
			endif
		endfor
		
		if not keyword_set(run_check) then begin
			z=sort(sall)
			s=sall(z)
			pix=pixall
			pix(*,0)=pix(z,0)
			pix(*,1)=pix(z,1)
		endif
	endelse

endif else begin

	if(n_elements(s) gt 1) then begin

		;dnuc should be zero as we shoudl be subtracting the counts
		;in get_ir using the rawnuc option
		;if you wanted to subtrcat a temperature then you need to 
		;disable the rawnuc option and remove the zeroing of dnuc
		;this is left in case at a later data a temperature does 
		;need to be subtracted. Eh?
		sval=fix(s*1000)/1000.0
		a=indgen(n_elements(sval)-1)
		b=a+1
		ns=where(sval(b)-sval(a) gt 0.0005) ;AT 16/06/11 0.001 to 0.0005
		ns=[ns,n_elements(sval)-1] ;not sure this properly accounts for 
					   ;all quantised s values

		snew=sval(ns)
		e=reverse((dim-dnuc),2)>0.0
		temp=[0]

		for is=0,n_elements(snew)-1 do begin
			ms=where(sval eq snew(is))
			temp=[temp,mean(e(pix(ms,0),pix(ms,1)))]
		endfor

		temp=temp(1:*)

		;stop
		;determine the number of saturated pixels looking at the raw data
		;and put a cut at 15000

		ee=reverse((rawd),2)>0.0
		rawl=ee(pix(*,0),pix(*,1))
		sat = where(rawl gt 15000,satpix)
	endif
		stemp=snew
endelse

end
