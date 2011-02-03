pro heat,sh,trange,ldef,t,s,h,qpro,numsatpix,alphaconst,disp=disp,print=print,tsmooth = tsmooth,tbgnd= tbgnd,targetrate=targetrate,nline=nline,aug=aug
;begin analysis - nuc =1 means average over and subtract frames before the start of the shot
;   
;leon,sh,trange,nuc=1,theo=ldef,t,s,h,err,numsatpix,disp=disp,targetrate=targetrate,nline=nline,aug=aug

;print,'number of time slices returned from leon',n_elements(t)

;if(err ne '') then begin
;	print,err
;	return
;endif

; parameters for EK98
diff=[80.,21.,15.]  ; mm^2/s
diff=diff*1.e-6     ; m^2/s
lam=[115.,63.,49.]  ; W/m/K
temp=[0,500,1000]   ; C
;diff=[100.8,91.4,48.6,28.5,18.8,14.4,11.1,8.8,7.4,6.0,4.8]  ; mm^2/s
;diff=diff*1.e-6     ; m^2/s
;lam=[82.,80.,68.,62.4,54.4,44.,36.,30.,26.,22.,18.]  ; W/m/K
;temp=[0.,20.,123.5,227.,477.,727.,977.,1272.,1477.,1727.,2270.]   ; C
if keyword_set(alpha) then begin
	alpha_top=alpha
endif else begin
	alpha_top= alphaconst ; W/m**2/K heat transmission at the front site, AUG = 144000.0
endelse
alpha_bot =  250.       ; W/m**2/K heat transmission at the back site
aniso=1.0               ; anisotropy D_y/D_x, y ist quer, x ist tief

heat_impact = SQRT(diff(1))/lam(1) ; m**2/W * K/s**-0.5

if (keyword_set(tsmooth)) then begin
   print,' '
   print,'smoothing over ',tsmooth,' time steps'
   print,' '
   data2 = h
   for i=0,n_elements(s)-1 do data2(i,*) = smooth(h(i,*),tsmooth)
   h = data2
endif
if (keyword_set(tbgnd)) then begin

    rmaxt = fltarr(n_elements(t))
    maxt = fltarr(n_elements(t))

    for i=0,n_elements(t)-1 do begin

        maxt(i) = max(h(*,i),imax)
        rmaxt(i) = s(imax)
    endfor
;
    rmax = max(s)
    rmin = min(s)
    rstep = (rmax-rmin)/10.
;
;   start shot looking at the far end of the divertor
;
     rup = rmax - rstep
     rlow = rmin + rstep

;
;    when the strike point comes on the surface then look for which end and swap to start later in the shot
;

     switchval = 40
     meantemp = fltarr(n_elements(t))
     a = where(t gt 0.1 and rmaxt gt (rmin + 7.*rstep) and maxt gt switchval,cc)
     switch_time = max(t)
     if (cc gt 0 ) then switch_time  = t(a(0))
     print,'switch found at time',switch_time,' max time',max(t)
;
;
     print,' '
     print,'subtracting mean temperature'
     print,' '
     for i=0,n_elements(t)-1 do begin

        if (t(i) lt switch_time) then begin
            r=where(s gt rup)
            meantemp(i) = mean(h(r,i))
        endif else begin
            r=where(s lt rlow)
            meantemp(i) = mean(h(r,i))
        endelse
;
;   subtract mean temp and add 20 back on i.e. the background
;
        h(*,i) = h(*,i) - meantemp(i) + 20.
      endfor

      h = h > 20.


endif
;FL: So we can run with several values of alpha, if alpha_top is an array
;    then we will run Theodor that many times and stick them together.
;    We can then put seperate them again in idaout.
n_alphas=n_elements(alpha_top)
qpro=theo_new(h,t,s,0.028,alpha_bot,alpha_top(0),$
              diff,lam,aniso,test=test,t_profiles=t_profiles,h_pot=h_pot)
if n_alphas gt 1 then begin
   for n=1,n_alphas-1 do begin
	qpron=theo_new(h,t,s,0.028,alpha_bot,alpha_top(n),$
              diff,lam,aniso,test=test,t_profiles=t_profiles,h_pot=h_pot)
	qpro=[[[qpro]],[[qpron]]]
   endfor
endif

toffset=get_toffset(sh);function earlier in this file
print,'Time offset:',toffset,'s'
t=t-toffset
end

