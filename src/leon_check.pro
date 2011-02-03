pro leon_check,sh,t,nuc=nuc,cal=cal,recal=recal,theo=theo,disp=disp,tall,stemp,alltemp,err,numsatpix,mid=mid,targetrate=targetrate,nline=nline,aug=aug

;program to check the alignment from leon and display the result

;first need to read the align.dat file to locate the anaylsis path chosen
;and then find it in the ldefs.dat file
;read from align
shot=long(sh)
alignarr=readtxtarray('align.dat', 3)
leshot=where(alignarr(0,*) le shot, noel)
loc=alignarr(1,n_elements(leshot)-1)
loc2=alignarr(2,n_elements(leshot)-1)

;convert location to path using ldef.dat
get_ldef, loc, theo

;check if this covers two strikepoints
if (loc2 ne '') then begin
	get_ldef, loc2, theo2
endif else theo2=-10

common data,d,width,height,xst,yst,comp,dnuc,pix,s
common image,bset,cset,bcut,tcut
common invessel,ribs,boxs,p2,cc
t(1)=t(1)+get_toffset(sh);FL: Make time window longer to account for offset
components,sh,ribs,boxs,p2,cc,/gdc,mid=mid ;passes through override to put MID in. Procedure in invessel.pro.

err=''

;determine if the shot is gt 10000 or lt 10000 and generate the path
;as required
ftype=size(sh,/type)
if(ftype ne 7) then begin
	shotstring = strtrim(string(sh),1)
	shotnr = sh
	shot_int1=shotnr/100
	shot_int2=shotnr-shot_int1*100
	shot_str=STRING(FORMAT='(i4.4, a1, i2.2)', shot_int1, '.', shot_int2)
	fname='rir'+shot_str
	filename='$MAST_DATA/'+shotstring+'/Images/rir0'+shotstring+'.ipx'
endif

;look for the calibration file in the /calib directory. If exact match not
;available, then used the calibation from the nearest previous shot.
if(ftype ne 7) then begin
    ;check existance of calibration file database
    get_lun,lun
    openr,lun,'calib/vecs.dat',error=err
    close,/all

    ;search for calibration file closest to shot number
    if(err eq 0) then begin
      	calsh=fix(readtxtarray('vecs.dat',1))
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

if !d.name ne 'PS' then window,0,xs=320*2,ys=256*2
!x.margin=[0,0] ;need these otherwise wireframe plotted in wrong place
!y.margin=[0,0]

maxtime=1.0 ;set upper limit on duration of analysis

;read in the IPX data from the file
dummy=readipx(sh,framerate,integtime,numframes,width,height,$
xst,yst,gain,offset,data,filename,time,maxtime,lambda,targetrate=targetrate)

;any end time can be entered when setting trange in iranalysis - the end time
;is set here, by the maximum time found in the IPX file
if (t(1) gt max(time)) then t(1) = max(time)

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
		;this is subsequently used in get_ir to subtract from the data
		rawnuc = rawd
		;dnuc is the temperature equivalent of the raw data - it used to be
		;used in get_trace - setting it to 0 means it is no longer used there
	    	dnuc=dnuc*0.0
	endif else begin
	    	get_ir,sh,time,data,t(0),d,integtime,cphot,framerate,rawd,lambda
	    	dnuc=d*0.0 ;gfc modification - is using time t(0) sensible?
	endelse

	yst=(256-(yst+n_elements(dnuc[0,*])))*2
	xst=xst*2
	bset=0.0
	cset=100.0
	bcut=-1e38
	tcut=1e38

	fname_i = 'calib/'+fname
	if (keyword_set(aug)) then fname_i = 'calib/'+fname
	fname = fname_i
	print,'Loading alignment file ',fname
	get_lun,lun
	openr,lun,fname,error=err
	close,/all

	;load the saved alignment file
	restore,fname,/verb
	ptco=(l(0)-o(0))/n(0)
	update_dist,ptco    

	;get frame in middle of time window to overplot the wireframe on
	get_ir,sh,time,data,(t(0)+t(n_elements(t)-1))/2.0,d,integtime,cphot $
	,framerate,rawd,lambda,rawnuc=rawnuc, /run_check

	set_display, /scon, w=0

	if(keyword_set(disp) or keyword_set(cal) or keyword_set(recal)) then begin
		!p.multi=0
		plot,[100],[100],xr=[-320,320]/2*30e-6,yr=[-256,256]/2*30e-6,xs=5,ys=5,/noerase
		;this draws the wireframe over the IR camera image plotted with
		;set_display, /scon, w=0
		ccd_image_at,o,p,i,j,n,l,ribs,boxs,p2,cc
endif

;See whether the analysis path is an area or a single line. 
n_theo=n_elements(theo)
if(n_theo ne 0) then begin
	if(n_theo eq 6) then get_trace,d,rawd,theo,temp,stemp,o,n,i,j,l,satpix,/line,disp=disp, /run_check
	if(n_theo eq 12) then get_trace,d,rawd,theo,temp,stemp,o,n,i,j,l,satpix,/area,disp=disp,nline=nline, /run_check
	if (keyword_set(disp)) then comp=tvrd()
endif

;check to see if there are two strikepoints - ie, theo2 is set
if theo2[0] ne -10 then begin

	n_theo=n_elements(theo2)
	if(n_theo ne 0) then begin
	if(n_theo eq 6) then get_trace,d,rawd,theo2,temp,stemp,o,n,i,j,l,satpix,/line,disp=disp, /run_check
	if(n_theo eq 12) then get_trace,d,rawd,theo2,temp,stemp,o,n,i,j,l,satpix,/area,disp=disp,nline=nline, /run_check
;	if (keyword_set(disp)) then comp=tvrd()
endif

endif

;print the shotnumber and analysis path name to the plot
xyouts, 0.05,0.90, strtrim(string(fix(shot)),2), /normal, charsize=1.75, color=truecolor('firebrick')
xyouts, 0.05,0.87, strtrim(loc,2), /normal, charsize=1.75, color=truecolor('firebrick')
if loc2 ne -10 then xyouts, 0.15, 0.84, strtrim(loc2,2), /normal, charsize=1.75, color=truecolor('firebrick')

end
