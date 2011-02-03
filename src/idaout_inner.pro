
	pro idaout_inner,fid,shotnr,view,time,scoord,tprofiles,qproarray,ldef,numsatpix,alphaconst,tc_state,scheduler=scheduler,aug=aug,inner=inner

@ida3



fshot=STRING(FORMAT='(i6.6)', shotnr)
fprefix = 'air'
;
;   time offset now handled in readimg_fast in heat.pro
;
;We could use a whole raft of alphas, b
	qprofiles=reform(qproarray(*,*,0))

;
   r_start = ldef[0,0]
   r_extent = ldef[0,1]
   phi_start = ldef[1,0]
   phi_extent = ldef[1,1]
   z_start = ldef[2,0]
   z_extent = ldef[2,1]
;
;    try to define if on centre column or rins
;    iconfig = 1 means on outboard rib
;    iconfig = 2 means on inboard rib
;    iconfig = 3 means on outer wall  
;    iconfig = 0 means unknown         
;
   iconfig = 0
   if (abs(r_extent) gt 0 and abs(z_extent) eq 0 ) then iconfig = 1
   if (abs(r_extent) eq 0 and abs(z_extent) gt 0 ) then iconfig = 2
   if (abs(r_extent) eq 0 and abs(z_extent) gt 0 and r_start gt 1.) then iconfig = 3
;iconfig=2
;
;   if we dont know what config we are in set radial coord as s coord
;
   rcoord = scoord
   n_scoord = n_elements(scoord)
   zcoord = replicate(0.,n_scoord)

;   if we are on the outboard rib then the radial coord = scoord + offset
;
   if (iconfig eq 1) then rcoord = scoord + r_start
   if (iconfig eq 1) then zcoord = replicate(z_start,n_scoord)
;
;   if we are on the inner rib then we need to make a radial array at the cc
;
   if (iconfig eq 2) then rcoord = replicate(r_start,n_scoord)
   if (iconfig eq 2) then zcoord = scoord + z_start
;print, scoord
;
;   if we are on the outer wall
;
   isign = 1
   if (z_extent lt 0 ) then isign = -1
   if (iconfig eq 3) then begin 
      rcoord = replicate(r_start,n_scoord)
      zcoord = isign*scoord + z_start
;
;     to be compatible with xpad the coords haveto increasing hence if isign
;     is negative we have to reverse the zcoord and reverse the tprofile
;      and qprofile
;
      if (isign lt 0) then begin
         zcoord = reverse(zcoord)
         tprofiles=reverse(tprofiles)
         qprofiles=reverse(qprofiles)
;
;     look on p5 upper
;
         a=where(zcoord gt 0.45 and zcoord lt 0.6)
         zcoord = zcoord(a)
         tprofiles = tprofiles(a,*)
         qprofiles = qprofiles(a,*)
      endif
   endif



; calculate power

; equally spaced scoord, delta scoord is constant
   max_times = n_elements(time)
   ;ds=scoord(1)-scoord(0);FL 29/7/2005 ds not constant for area calcs
   dt=time(1)-time(0)
   powertmp=0
   power=fltarr(max_times)
   energy=fltarr(max_times)
   
;   for i=0l,max_times-1 do begin
;      power(i)=total(qprofiles(*,i)*rcoord(*)*2*3.1415926*ds)
;   endfor
; NEW VERSION AS FOLLOWS 3/8/2005
   for i=0l,max_times-1 do begin
      power(i)=0.5*total((qprofiles(1:n_scoord-1,i)+qprofiles(0:n_scoord-2,i))*!pi*(rcoord(1:n_scoord-1)+rcoord(0:n_scoord-2))*(scoord(1:n_scoord-1)-scoord(0:n_scoord-2)))

      ;Now add end ds;x FL 3/8/2005 Not actually necessary      
;      power(i)=power(i)+(qprofiles(n_scoord-1,i)*2*!pi*rcoord(n_scoord-1)*(scoord(n_scoord-1)-scoord(n_scoord-2)))
   endfor
print,0.5*(rcoord(1:n_scoord-1)+rcoord(0:n_scoord-2));*(scoord(1:n_scoord-1)-scoord(0:n_scoord-2))

   energy(0)=0
   for i=1l,max_times-1 do begin
      energy(i)=energy(i-1)+power(i)*dt
   endfor

        
   power=power/1e6
   energy=energy/1e6
   qprofiles=qprofiles/1e6   

   nelem=1
   s=intarr(nelem)
   s(0)=view   
;
;  write out the configuraration
;
    item = ida_create(fid, fprefix+'_CAMERA VIEW_ISP', fshot)
    ok = ida_set_structure(item, ida_dct, 1, ida_d4+ida_intg, xtsams=nelem)
    ok = ida_set_dinfo(item, 0.0, 1.0, 0, 1.0, 0.0, 'view', 'view')
    ok = ida_set_data(item, s, ida_d4+ida_intg+ida_valu)
    ok = ida_free(item)
;
;  write out the value of alpha used
;
   nelem=1
   s=fltarr(nelem)
   s(0)=alphaconst(0)  
   item = ida_create(fid, fprefix+'_ALPHACONST_isp', fshot)
   ok = ida_set_structure(item, ida_dct, 1, ida_d4+ida_real, xtsams=nelem)
   ok = ida_set_dinfo(item, 0.0, 1.0, 0, 1.0, 0.0, 'alpha', 'alpha')
   ok = ida_set_data(item, s, ida_d4+ida_real+ida_valu)
;
;  write out the rstart         
;
   nelem=1
   s=fltarr(nelem)
   s(0)=r_start
   item = ida_create(fid, fprefix+'_R_START_isp', fshot)
   ok = ida_set_structure(item, ida_dct, 1, ida_d4+ida_real, xtsams=nelem)
   ok = ida_set_dinfo(item, 0.0, 1.0, 0, 1.0, 0.0, 'm','radius')
   ok = ida_set_data(item, s, ida_d4+ida_real+ida_valu)
   ok = ida_free(item)
	
;
;  write out the phistart         
;
   nelem=1
   s=fltarr(nelem)
   s(0)=phi_start
   item = ida_create(fid, fprefix+'_PHI_START_isp', fshot)
   ok = ida_set_structure(item, ida_dct, 1, ida_d4+ida_real, xtsams=nelem)
   ok = ida_set_dinfo(item, 0.0, 1.0, 0, 1.0, 0.0, 'deg','phi')
   ok = ida_set_data(item, s, ida_d4+ida_real+ida_valu)
   ok = ida_free(item)
	
;
;  write out the zstart         
;
   nelem=1
   s=fltarr(nelem)
   s(0)=z_start
   item = ida_create(fid, fprefix+'_Z_START_isp', fshot)
   ok = ida_set_structure(item, ida_dct, 1, ida_d4+ida_real, xtsams=nelem)
   ok = ida_set_dinfo(item, 0.0, 1.0, 0, 1.0, 0.0, 'm','Z')
   ok = ida_set_data(item, s, ida_d4+ida_real+ida_valu)
   ok = ida_free(item)
	
;
;  write out the rextent        
;
   nelem=1
   s=fltarr(nelem)
   s(0)=r_extent
   item = ida_create(fid, fprefix+'_R_EXTENT_isp', fshot)
   ok = ida_set_structure(item, ida_dct, 1, ida_d4+ida_real, xtsams=nelem)
   ok = ida_set_dinfo(item, 0.0, 1.0, 0, 1.0, 0.0, 'm','radius')
   ok = ida_set_data(item, s, ida_d4+ida_real+ida_valu)
   ok = ida_free(item)
	
   nelem=1
   s=fltarr(nelem)
   s(0)=phi_extent
   item = ida_create(fid, fprefix+'_PHI_EXTENT_isp', fshot)
   ok = ida_set_structure(item, ida_dct, 1, ida_d4+ida_real, xtsams=nelem)
   ok = ida_set_dinfo(item, 0.0, 1.0, 0, 1.0, 0.0, 'deg','phi')
   ok = ida_set_data(item, s, ida_d4+ida_real+ida_valu)
   ok = ida_free(item)
	
   nelem=1
   s=fltarr(nelem)
   s(0)=z_extent
   item = ida_create(fid, fprefix+'_Z_EXTENT_isp', fshot)
   ok = ida_set_structure(item, ida_dct, 1, ida_d4+ida_real, xtsams=nelem)
   ok = ida_set_dinfo(item, 0.0, 1.0, 0, 1.0, 0.0, 'm','Z')
   ok = ida_set_data(item, s, ida_d4+ida_real+ida_valu)
   ok = ida_free(item)
	
; Generate some useful output, max(data), profiles at x seconds
   rqmax=fltarr(max_times)
   qmax=fltarr(max_times)
   lamsol=fltarr(max_times)
   lampp=fltarr(max_times)
   lamsol2=fltarr(max_times)
   lampp2=fltarr(max_times)
   tmin=fltarr(max_times)
   qmin=fltarr(max_times)
   rfft=fltarr(max_times)
;   ptot=fltarr(max_times)
   etot=fltarr(max_times)
   etotsum=fltarr(max_times)
   
        
   rcnew=reform(rcoord);FL Added for easier comparisons (lamsol etc)
   zcnew=reform(zcoord)

;FL 29/7/2005:Turns out PTOT can be done in 1 line, why did no-one tell me?
 ptot=power  
;ptot=power>0.0
        
   for i=1l,max_times-1 do begin
      qmax(i)=max(qprofiles(*,i),tmploc,MIN=tmp,/NAN)	
;
      rqmax(i)=rcoord(tmploc)
      if (iconfig eq 1 ) then rqmax(i)=rcoord(tmploc)
      if (iconfig eq 2 ) then rqmax(i)=zcoord(tmploc)
      if (iconfig eq 3 ) then rqmax(i)=zcoord(tmploc)
;
;FL 24/11/2003: Added this section to calculate PTOT, integrated positive power
;      posp=where(qprofiles(*,i) ge 0,noel)
;      if noel ne 0 then begin
;        if (iconfig eq 1 ) then begin
;          ptot(i)=2*!pi*total(qprofiles(posp,i)*rcoord(posp)*ds)
;        endif else ptot(i)=2*!pi*total(qprofiles(posp,i)*zcoord(posp)*ds)
;      endif else ptot(i)=0
      etot(i)=ptot(i)*dt; FL: ETOT is instantaneous energy
      etotsum(i)=total(etot(1:i))

      peak=where(qprofiles(*,i) eq qmax(i),noel)
      lam=where(qprofiles(*,i) ge qmax(i)/exp(1),noel2);OLD measure of lambda
      tail=n_elements(rcoord)-1

      if (noel ge 1) and (peak(0) ge 4) and (tail-peak(0) ge 4) then begin

          if (iconfig eq 1) then begin

            lampp(i)=interpol(rcnew(peak(0):tail),qprofiles(peak(0):tail,i),qmax(i)/exp(1))-rqmax(i)
             lamsol(i)=rqmax(i)-interpol(rcnew(0:peak(0)),qprofiles(0:peak(0),i),qmax(i)/exp(1))

            
          endif else begin

             lamsol(i)=interpol(zcnew(peak(0):tail),qprofiles(peak(0):tail,i),qmax(i)/exp(1))-rqmax(i)
             lampp(i)=rqmax(i)-interpol(zcnew(0:peak(0)),qprofiles(0:peak(0),i),qmax(i)/exp(1))
            
          endelse

      endif else begin

         lamsol(i)=0
         lampp(i)=0

      endelse
        
      if noel2 ne 0 then begin ;Output OLD measure of lambda

         if (iconfig eq 1) then begin

            lamsol2(i)=rcoord(lam(noel2-1))-rqmax(i)
            lampp2(i)=rqmax(i)-rcoord(lam(0))

          endif else begin

            lampp2(i)=rqmax(i)-zcoord(lam(noel2-1))
            lamsol2(i)=zcoord(lam(0))-rqmax(i)

          endelse

      endif else begin

          lamsol2(i)=0
          lampp2(i)=0

      endelse
          
        
      qmin(i)=min(qprofiles(*,i),/NAN); FL changed from tmp

   endfor	

; time evolution of maxima/minima
   tout = time(*) 

   sti = 0
   eti = max_times-1

   err = GC_PUTIT(fid, shotnr, fprefix+'_minpower_density_isp', 'kWm-2',$
               tout, 1.e3*qmin(sti:eti))
;
   err = GC_PUTIT(fid, shotnr, fprefix+'_pkpower_density_isp', 'kWm-2',$
                       tout, 1.e3*qmax(sti:eti))
;
;
;
   err = GC_PUTIT(fid, shotnr, fprefix+'_pkpower_pos_isp', 'm',$
                       tout, rqmax(sti:eti))
;
;
;   use the simpler methos as more reliable - really need to fit the profiles
;
   err = GC_PUTIT(fid, shotnr, fprefix+'_lampowpp_isp', 'm',$
                       tout, lampp2(sti:eti))
;
   err = GC_PUTIT(fid, shotnr, fprefix+'_lampowsol_isp', 'm',$
                       tout, lamsol2(sti:eti))
;         

   rout = reform(scoord)
   if (iconfig eq 1) then rout = reform(rcoord)
   if (iconfig eq 2) then rout = reform(zcoord)
   if (iconfig eq 3) then rout = reform(zcoord)
;
;   need to write this out as dcx not dcz as too many time domains
;   therefore need to make uniformly spaced in r
;
    diff = rout - shift(rout,1)
    rout_norm = rout
;
;    check to see if it needs resplining
;
    if (stddev(diff) gt 1.e-6) then begin
       print,'reinterpolating rout onto uniform space',stddev(diff)
;
;    exclude first point as is offset
;
       mean_diff = mean(diff(1:*))
       new_r = rout(0) + mean_diff*findgen(n_elements(rout))
       print,'mean difference between new and old',mean(new_r-rout)
       print,'largest difference between new and old',max(abs(new_r-rout),im)
       print,'old value ',rout(im),' new value ',new_r(im)
       rout_norm = new_r


     endif

;
   err = put_trace_dt(fid,shotnr,fprefix+'_qprofile_isp','kW m-2', $ 
           'IR power profiles',rout_norm,tout,qprofiles)
;
   err = put_trace_dt(fid,shotnr,fprefix+'_tprofile_isp','Degree C', $
           'Temperature profiles',rout_norm,tout,tprofiles) 


   err = GC_PUTIT(fid, shotnr, fprefix+'_satpixels_isp', 'number',$ 
 	tout, numsatpix)


;
;        because IDA can not take a none uniform radius also write out the radius
;
        
    err = put_trace_dt(fid,shotnr,fprefix+'_rcoord_isp','m','IR true coord',rout,rout,/no_t_base)
;


; energies


;FL 24/11/2003: adding PTOT and ETOT trace
    err = GC_PUTIT(fid, shotnr, fprefix+'_ptot_isp', 'kW',$ 
                   tout, 1.e3*ptot(sti:eti))
    err = GC_PUTIT(fid, shotnr, fprefix+'_etot_isp', 'kJ',$ 
                   tout, 1.e3*etot(sti:eti))
    err = GC_PUTIT(fid, shotnr, fprefix+'_etotsum_isp', 'kJ',$ 
                   tout, 1.e3*etotsum(sti:eti))
      
; Some Info about the TEMPERATURE


; time evolution of maxima/minima

   maxtemp=fltarr(max_times)

   for i=0l,max_times-1 do begin
      maxtemp(i)=max(tprofiles(*,i))
   endfor

   err = GC_PUTIT(fid, shotnr, fprefix+'_temperature_isp', 'Degree C',$ 
                       tout, maxtemp(sti:eti))

SKIP:

end

