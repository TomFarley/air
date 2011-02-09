pro leon,sh,t,nuc=nuc,cal=cal,recal=recal,theo=theo,disp=disp,tall,stemp,alltemp,err,numsatpix,mid=mid,targetrate=targetrate,nline=nline,aug=aug

PREFR = 'rir'

;LEON aligns the camera image with the wireframe model of MAST
;theo - is the analysis path (read in from ldef if iranalysis)

common data,d,width,height,xst,yst,comp,dnuc,pix,s
common image,bset,cset,bcut,tcut
common invessel,ribs,boxs,p2,cc
t(1)=t(1)+get_toffset(sh);FL: Make time window longer to account for offset
components,sh,ribs,boxs,p2,cc,/gdc,mid=mid ;passes through override to put MID in. Procedure in invessel.pro.

;set up display if plotting
if (keyword_set(disp)) then begin
    reds=  bytscl([1, 0, 1, 0, 0, 1.0, 0, 0.75])
    greens=bytscl([1, 0, 0, 1, 0, 1.0, 0.75, 0])
    blues= bytscl([1, 0, 0, 0, 1, 0., 0.75, 0.75])
    tvlct, reds, greens, blues
   device,decompose=0
   !p.color=1
;  loadct,0
endif
err=''

;determine if the shot is gt 10000 or lt 10000 and generate the path
;as required
ftype=size(sh,/type)
if(ftype ne 7) then begin
   shotnr = strtrim(string(sh),1)
   k_shot = str(shotnr/1000,formstat='(i3.3)')
   i_shot = str(shotnr,formstat='(i6.6)')
   fname  = PREFR+i_shot+'.ipx'

   filename = getidampath(shotnr,PREFR)
   type_check = size(filename,/type)
   print,'type_check = ',type_check

   if (type_check eq 3) then begin
; Type returned is LONG - ie ERROR
     filename = '$MAST_DATA'+'/'+k_shot+'/'+shotnr+'/'+'LATEST'+'/'+fname
     if (keyword_set(aug)) then begin
       filename='$MAST_IMAGES'+'/'+PREFR+'/'+fname
     endif
     print,'RAW FILE USED [',filename,']'
   endif else begin
     print,'IDAM FILE OK [',filename,'] - Continue...'
   endelse

    get_lun,lun
        openr,lun,filename,error=err
        if(err eq 0) then begin
                close,/all
                free_lun,lun
        endif else begin
                err=filename+' not found'
                free_lun,lun
                return
        endelse
endif else begin
	filename='$MAST_IMAGES/rir/'+sh
endelse

;look for the calibration file in the /calib directory. If exact match not
;available, then used the calibation from the nearest previous shot.
if(ftype ne 7) then begin
    ;check existance of calibration file database
    get_lun,lun
    openr,lun,'calib/vecs.dat',error=err
    close,/all

    ;search for calibration file closest to shot number
    if(err eq 0) then begin
      	calsh=fix(readtxtarray('vecs.dat',1)) ;FL:  This reads in single column from vecs.dat
      	q=where(calsh le sh)
      	if(q(0) eq -1) then begin
        	print,'Shot not found in alignment database - using vectors.dat'
	        fname='vectors.dat'
	endif
      	fname=strtrim(string(calsh(q(n_elements(q)-1))),2)+'cal.dat'
    endif else begin
	print,'no vecs.dat file'
	if(ftype ne 7) then begin
        	fname=strmid(sh,0,strlen(a)-4)+'cal.dat'
        endif else begin
  	      	print,'No alignment file found - using vectors.dat'
        	fname='vectors.dat'
        endelse
    endelse
endif else begin
    fname=strmid(sh,0,strlen(sh)-4)+'cal.dat'
endelse

;setup for plotting, again.
if(keyword_set(disp) or keyword_set(cal) or keyword_set(recal)) then begin
	window,0,xs=320*2,ys=256*2
	!x.margin=[0,0]
	!y.margin=[0,0]
	if(keyword_set(theo)) then window,1,xs=200,ys=200
	wset,0
endif

maxtime=1.0 ;set upper limit on duration of analysis

;read in the IPX data from the file
dummy=readipx(sh,framerate,integtime,numframes,width,height,$
xst,yst,gain,offset,data,filename,time,maxtime,lambda,targetrate=targetrate)

;any end time can be entered when setting trange in iranalysis - the end time
;is set here, by the maximum time found in the IPX file
if (t(1) gt max(time)) then t(1) = max(time)

lambda=readheadfile(sh,/lambda,aug=aug);This gets the wavelength from the txt file
print,'Filter:',lambda

;perform NUC using first frame of the discharge
if(n_elements(nuc) ne 0) then begin
	print,'Perform NUC'

	;try to nuc from first frame to 1 ms before the shot
	;otherwise just use the first frame
	tmin = min(time)
	tmax = -1.e-3
	nuc = [tmin,tmax]

	if (tmin gt tmax) then nuc = t(0)
	        ;obtain frame at nuc time
		get_ir,sh,time,data,nuc,dnuc,integtime,cphot,framerate,rawd,lambda
		;use raw info to set nuc
		;this is subsequently used in get_ir to subtract form the data
		rawnuc = rawd
		;dnuc is the temperature equivalent of the raw data - it used to be
		;used in get_trace - setting it to 0 means it is no longer used there
	    	dnuc=dnuc*0.0
	endif else begin
	    	get_ir,sh,time,data,t(0),d,integtime,cphot,framerate,rawd,lambda
	    	dnuc=d*0.0 ;gfc modification - is using time t(0) sensible?
	endelse

	yst=(256-(yst+n_elements(dnuc(0,*))))*2 ;y offset - dont know why dnuc
;						is involved (breaks if
						;removed). Definition is also
						;used in get_line.pro
	xst=xst*2 ;x offset - image is rebinned to twice orig. size in
		  ;set_display, hence the *2
	bset=0.0 ;set the brightness top level
	cset=100.0 ;set the lower level
	bcut=-1e38
	tcut=1e38

	;load the calibration file
	fname_i = 'calib/'+fname
	if (keyword_set(aug)) then fname_i = 'calib/'+fname
	fname = fname_i
	print,'Loading alignment file ',fname
	get_lun,lun
	openr,lun,fname,error=err
	close,/all

	if (err eq 0 and (not keyword_set(cal))) then begin
;.... MODIFIED code to use SAVE file extraction of HEADER info.
	  sObj = OBJ_NEW('IDL_Savefile' ,fname)
	  sContents = sObj->Contents()
	  print ,"Calib file [",fname,"]"
	  print ,"Created : [",sContents.user,"-",sContents.date ,"]"
	  restore ,fname


		;********************* GFC ***********************
		ptco=(l(0)-o(0))/n(0)
		update_dist,ptco              ;FL:updates distortion common block
	endif else begin
		if(not keyword_set(cal) or keyword_set(recal)) then begin
			print,'Error - no ',fname,' file'
			return
		endif else begin
			get_ir,sh,time,data,t,d,integtime,cphot,framerate,rawd,lambda
			display ;adjust the contrast and replot image
			vectors,ptco,ro,phio,zo,alpha,beta,gamma                
			define_ccd,ptco,ro,phio,zo,alpha,beta,gamma,o,p,i,j,n,l,err
		endelse
	endelse

	if(keyword_set(disp) or keyword_set(cal) or keyword_set(recal)) then begin
		!p.multi=0
		plot,[100],[100],xr=[-320,320]/2*30e-6,yr=[-256,256]/2*30e-6,xs=5,ys=5,/noerase

		;this draws the wireframe
		ccd_image,o,p,i,j,n,l,ribs,boxs,p2,cc

	if (keyword_set(disp)) then comp=tvrd()
endif

;this part performs the alignment of the image and the wireframe
if(keyword_set(cal) or keyword_set(recal)) then begin
	get_ir,sh,time,data,t,d,integtime,cphot,framerate,rawd,lambda
	bb='A'
        img=[0,0,0]
        obj=[0,0]
        f=[0,0,0]
	while(strupcase(bb) ne 'S' and strupcase(bb) ne 'E') do begin
                if(strupcase(bb) eq 'N') then begin
                  img=[0,0,0]
                  obj=[0,0]
                  f=[0,0,0]
                endif
                if(strupcase(bb) eq 'W') then img(2,*)=img(2,*)
		if(strupcase(bb) eq 'A' or strupcase(bb) eq 'W') then begin
                  specify_point,o,p,i,j,n,l,ribs,boxs,p2,cc,img,obj,f
                endif
		if(strupcase(bb) eq 'R' or strupcase(bb) eq 'A' or $
                   strupcase(bb) eq 'N' or strupcase(bb) eq 'W') then begin
                  close=0.01
		;********************* GFC ***********************
                  if(n_elements(f) ne 1) then begin
                    find_vectors,ptco,o,p,i,j,n,l,ribs,boxs,p2,cc,f,obj,img,close,/new,/disp
                    print,'closeness is ',close
                    set_display,w=0
                    frame
                  endif
                endif
		if(strupcase(bb) eq 'C') then begin
                  close=0.001
		;********************* GFC ***********************
                  if(n_elements(f) ne 1) then begin
                    find_vectors,ptco,o,p,i,j,n,l,ribs,boxs,p2,cc,f,obj,img,close,/disp
                    print,'closeness is ',close
                    set_display,w=0
                    frame
                  endif
                endif
		print,'Options:'
                print,'Continue to finer search'
                print,'Restart with same points'
                print,'New points and restart'
                print,'Add points and restart'
                print,'Weight points, add new ones and restart'
                print,'Save calibration file and end'
                print,'End'
                read,bb
        endwhile
        if(strupcase(bb) eq 'S') then begin
              save,filename=fname,o,p,i,j,n,l
        endif
	return
endif

;See whether the analysis path is an area or a single line. 
n_theo=n_elements(theo)
if(n_theo ne 0) then begin
	if(n_theo eq 6) then get_trace,d,rawd,theo,temp,stemp,o,n,i,j,l,satpix,/line,disp=disp
	if(n_theo eq 12) then get_trace,d,rawd,theo,temp,stemp,o,n,i,j,l,satpix,/area,disp=disp,nline=nline
	if (keyword_set(disp)) then comp=tvrd()
endif

;get frame in middle of time window, for displaying if /disp set?
get_ir,sh,time,data,(t(0)+t(n_elements(t)-1))/2.0,d,integtime,cphot $
	,framerate,rawd,lambda,rawnuc=rawnuc 

if(keyword_set(disp)) then display

alltemp=0
tall=[0.0]
j1=0
for tr=t(0),t(n_elements(t)-1),1.0/framerate do begin
	j1 = j1 + 1
	tset=tr
	get_ir,sh,time,data,tset,d,integtime,cphot,framerate $
		,rawd,lambda,rawnuc=rawnuc
	if(keyword_set(disp)) then set_display,w=1
	if(n_elements(theo) ne 0) then begin
		get_trace,d,rawd,theo,temp,stemp,o,n,i,j,l,satpix
		if(keyword_set(disp)) then begin
			if(n_elements(stemp) gt 1) then begin
				wset,1
				plot,stemp,temp,xmargin=[5,2],ymargin=[2,2],yrange=[0,100],$
										chars=1.2
				wset,0
			endif
		endif
		if(n_elements(stemp) gt 1) then begin
			if(n_elements(alltemp) eq 1) then begin
				alltemp=temp
				tall=tset
                               	numsatpix = satpix
			endif else begin
				alltemp=[[alltemp],[temp]]
				tall=[tall,tset]
                               	numsatpix = [numsatpix,satpix]
			endelse
		endif
	endif
endfor

snew=findgen(n_elements(s)*2)/(n_elements(s)*2-1)*(s(n_elements(s)-1)-s(0))+s(0)
alltempn=fltarr(n_elements(snew),n_elements(alltemp(0,*)))

for itall=0,n_elements(alltemp(0,*))-1 do begin
	alltempn(*,itall)=spline(s,alltemp(*,itall),snew)
endfor

end
