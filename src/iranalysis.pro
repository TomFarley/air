pro iranalysis,sh,trange,loc,t,s,h,qpro,ldef,numsatpix,alphaconst,disp=disp,print=print,tsmooth=tsmooth,tbgnd=tbgnd,targetrate=targetrate,nline=nline,aug=aug

;variables
;sh = shot
; trange= time range, array e.g [0.0,0.1]
;ldef - returned from get_ldef, which uses loc
;loc - analysis path name e.g. 'louvre4'
;t - returned times
;s - returned radius
;h - returned temperature?
;qpro - returned heat flux
;numsatpix - total number of saturated pixels on analysis path
;alphaconst - alpha constant, can be array, defined in rit2air_comb_view.pro
;tsmooth - 
;tbgnd - 
;targetrate - 
;nline - 
;aug - run for ASDEX, defunct? 
;print - set flag for output to PS

;find the analysis path to be used (loc) in the ldef path definitions file
get_ldef,loc,ldef 

;run leon to convert to radius and convert counts to temperature
leon,sh,trange,nuc=1,theo=ldef,t,s,h,err,numsatpix,disp=disp,targetrate=targetrate,nline=nline,aug=aug

;check there is something to analyse from LEON
print,'Number of time slices returned from LEON',n_elements(t)
if(err ne '') then begin
	print,'No data returned from LEON'
	stop
endif

;run the heat flux analysis code THEODOR to get the heat flux
heat,sh,trange,ldef,t,s,h,qpro,numsatpix,alphaconst,disp=disp,print=print,tsmooth=tsmooth,tbgnd=tbgnd,targetrate=targetrate,nline=nline,aug=aug

; I have added these two line which were previously in infra (AK 18/11/03)
if keyword_set(disp) then begin
	device,retain=2
	device,decompose=0
endif

;setup the display if plotting to screen (/disp set)
if (keyword_set(disp)) then begin 
	loadct,5
   window,xs=400,ys=400   ; not active
   !x.style=1
   !y.style=1
   !x.range=0
   !y.range=0
   !x.margin=[5,5]
   !y.margin=[5,5]
   !p.multi=0
   !p.charsize=1.5
   
   ;plot heat flux profile
   shade_surf,qpro,s,t,az=0,ax=90,$
   shades=bytscl(qpro,top=!D.Table_size),$
   background=0,col=255,zcharsize=0.00001,charsize=1.5
endif

   ;looks like calculation of the divertor area?
   aind=indgen(n_elements(s)-1)
   bind=aind+1
   das2=s(bind)-s(aind)
   das=2.0*!pi*(s+ldef(0,0))*das2 ;why not 1/2 as one Rib group only?

   qtot=fltarr(n_elements(t)) ;total heat flux at one time
   qpeak=fltarr(n_elements(t)) ;peak heat flux at one time
   qwid=fltarr(n_elements(t)) ;heat flux width?
   rpeak=fltarr(n_elements(t)) ;radius where peak q occurs
   gpro=qpro*0.0 ;
   print,'number of time slices',n_elements(t)
   for i=0,n_elements(t)-1 do begin
	qtot(i)=total(qpro(*,i)*das)
	qpeak(i)=max(qpro(*,i))
	q=where(qpro(*,i) eq max(qpro(*,i)))
	rpeak(i)=s(q(0))
	qps=qpro(*,i)
	q=where(abs(s-rpeak(i)) le 0.2)
	qps=qps(q)
	qps=smooth(qps,5)
	q=where(qps ge max(qps)/2.0)  
	if(q(0) ne -1) then begin
		qwid(i)=s(q(n_elements(q)-1))-s(q(0))
	endif else begin
		qwid(i)=0.0
	endelse
	gpro(*,i)=total(qpro(*,i)*das2)/sqrt(2.0*!pi)/qwid(i)*$
	exp(-1.0/2.0*(s-rpeak(i))^2/qwid(i)^2)
   endfor

;plot the output (if /disp set)
if (keyword_set(disp)) then begin 
   window,2,xs=350,ys=700
   !x.style=1
   !y.style=1
   !x.range=[0.14,0.345]
   !y.range=0
   !x.margin=[10,5]
   !y.margin=[5,5]
   !p.multi=[0,0,4]
   plot,t,rpeak+ldef(0,0),yr=[ldef(0,0),ldef(0,0)+ldef(0,1)],ys=1,xs=1
   plot,t,qtot,xs=1
   plot,t,qpeak,xs=1
   plot,t,qwid,xs=1
endif

;output plots to postscript (if /print set)
if(keyword_set(print))then begin
	fname=strtrim(string(sh),2)+loc+'.eps'
	set_plot,'ps'
	device,filename=fname,xsize=15,ysize=20,yoffset=5,/encapsulate
	!x.margin=[10,5]
	!y.margin=[0,5]
	!p.multi=[0,0,4]
	!p.charsize=1
	plot,t,rpeak+ldef(0,0),yr=[ldef(0,0),ldef(0,0)+ldef(0,1)],ys=1,xs=1
	plot,t,qtot,yr=[0,4000000],xs=1
	plot,t,qpeak,yr=[0,8000000],xs=1
	plot,t,qwid,xs=1
	device,/close
	set_plot,'x'
endif

end
