pro rir2air_comb_view,shotnum=shotnum,pn=pn

;
;   error flag: -1 can not open raw file
;               -2  can not determine whether upper or lower view
;               -3  can not determine whether inboard and/or outboard
;		-4 The time base of the data is incorrect.
;               10+ inboard target
;              100+ outboard target
;

 iflag = 0
 isched = 0

;
;   check if pn keyword set  - if set then assume called by scheduler
;   if not then set to zero and assume local
;    
if (exists(pn) eq 1 ) then begin
	isched = 1
endif else begin
      pn=0
endelse

;find out if the IDA file exists
idaout_open,shot=shotnum,pn=pn,fid

if (fid le 0 ) then print,'can not open output file'
if (fid le 0 ) then goto,SKIP                          
     
if (not keyword_set(shotnum)) then print,'no shotnumber'
if (not keyword_set(shotnum)) then stop                  

;trick to check is passnumber set - need this in case pn = 0 
ans = n_elements(pn) gt 0
if (ans eq 0 ) then print,'no passnumber'
if (ans eq 0 ) then stop                  


;    find out which target we are looking at
     outer = 0
     inner = 0

     PREFR = 'rir'
if (isched eq 0) then  $ ;isched is 0 if passnumber is not set (local)
     ir_raw = '$MAST_IMAGES/'+str(shotnum/1000,formstat='(i3.3)')+'/'+str(shotnum)+'/'+PREFR+str(shotnum,FORMSTAT='(i6.6)')+'.ipx'
if (isched eq 1) then  $ ;this is for the scheduler version
     ir_raw = '$MAST_IMAGES/'+str(shotnum/1000,formstat='(i3.3)')+'/'+str(shotnum)+'/'+PREFR+str(shotnum, FORMSTAT='(i6.6)')+'.ipx'
     
a=ipx_open(ir_raw) ;read in the data
ns = size(a)

if (ns(0) eq 0 ) then iflag = -1 
if (ns(0) eq 0 ) then goto,SKIP

view = a.header.view
print, 'view', view

;work out which camera view we are using
camera_view = 0
if (strpos(strupcase(view),'UPPER') ge 0 ) then camera_view = 3
if (strpos(strupcase(view),'LOWER') ge 0 ) then camera_view = 4
if (strpos(strupcase(view),'HL01') ge 0 ) then camera_view = 4
if (strpos(strupcase(view),'HL04') ge 0 ) then camera_view = 4

if (camera_view eq 0 ) then begin 
	print,' unknown view ',view
	iflag = -2
        goto, SKIP
endif
    

if (strpos(strupcase(view),'#') ge 0 ) then begin 
;     #1 means outer + inner #4 means outer only
        sv = strsplit(view,'#',/extract)
        sloc = strmid(sv(1),0,1)   ; sometimes it is written as #1" and the " causes problems
        if (sloc  eq '4' or sloc  eq '1' or sloc eq '6') then outer = 1
;AT mods for incorrect IPX headers 24/05/11
        if (sloc  eq '1') then begin
		inner = 1
	endif else begin
		;sometimes the header is not correct, and the view
		;covers the inner and the outer
		;if there is an ISP analysis line defined in the
		;align file then set inner=1
		alignarr=readtxtarray('align.dat',3)
		leshot=where(alignarr(0,*) le shotnum,noel)
		tmp=max(fix(alignarr(0,leshot)),loc)
		alignment=alignarr(2,loc)
		if alignment eq '' then begin
			inner=0
		endif else begin
			print, 'ISP analysis path will be run: IPX header incorrect'
			inner=1
		endelse

	endelse
;AT incorrect header mods end.
        if (inner+outer eq 0 ) then begin 
            print,' cannot work out view',view
            iflag = -3
            goto,SKIP                           
        endif
     
endif else begin ; if no info assume outer target only
       outer = 1
endelse
 
if (strpos(strupcase(view),'VIEW1') ge 0 ) then begin 
	outer = 1
	inner = 1
endif

print,' outer = ', outer, ' inner = ',inner

;run for outer target
if (outer eq 1) then begin
     alignarr=readtxtarray('align.dat',2) ;Read in from two column data file
     leshot=where(alignarr(0,*) le shotnum,noel)

;load the alignment from the calib directory
if noel ne 0 then begin
    print,'Using alignment set on shot',max(fix(alignarr(0,leshot)),loc)
    alignment=alignarr(1,loc)
endif else begin
    print,'Having trouble finding the correct alignment'
    stop
endelse

;set the outer target alpha constants
if (camera_view eq 3) then begin
      iview = 3
      alphaconst = [70000,30000]
endif
if (camera_view eq 4) then begin
      iview = 4
      alphaconst = [70000,30000]
endif
print,'using alpha = ',alphaconst

   startime = 0.0   
   endtime = 0.6  
;
print,'Processing shotnum ',shotnum,' Pass ',pn,' alignment ',alignment
;
      iranalysis,shotnum,[startime,endtime],alignment,t,s,temperature,power, $
         ldef,numsatpix,alphaconst,tsmooth=tsmooth,tbgnd=tbgnd, $
	 targetrate=targetrate,nline=nline


   r_start = ldef[0,0]
   r_extent = ldef[0,1]
   phi_start = ldef[1,0]
   phi_extent = ldef[1,1]
   z_start = ldef[2,0]
   z_extent = ldef[2,1]

IDAOUT:

    idaout_outer,fid,shotnum,iview,t,s,temperature,power,ldef,numsatpix,$
           alphaconst,tc_state,/scheduler
    iflag = iflag + 100
endif
;end of outer target analysis

;run for the inner target, if present
if (inner eq 1) then begin
     alignarr=readtxtarray('align.dat',3) ;Read in from two column data file
    leshot=where(alignarr(0,*) le shotnum,noel)
  if noel ne 0 then begin
    print,'Using alignment set on shot',max(fix(alignarr(0,leshot)),loc)
    alignment=alignarr(2,loc)
	if alignment eq '' then goto, SKIP
  endif else begin
    print,'Having trouble finding the correct alignment'
    stop
  endelse

;set inner target alpha values
   if (camera_view eq 3) then begin
      iview = 1
      alphaconst = [70000,30000]
   endif
   if (camera_view eq 4) then begin
      iview = 2
      alphaconst = [70000,30000]
   endif
   print,'using alpha = ',alphaconst

   startime = 0.0   
   endtime = 0.7  
;
   print,'Processing shotnum ',shotnum,' Pass ',pn,' alignment ',alignment
;
      iranalysis,shotnum,[startime,endtime],alignment,t,s,temperature,power, $
         ldef,numsatpix,alphaconst,tsmooth=tsmooth,tbgnd=tbgnd, $
	 targetrate=targetrate,nline=nline


   r_start = ldef[0,0]
   r_extent = ldef[0,1]
   phi_start = ldef[1,0]
   phi_extent = ldef[1,1]
   z_start = ldef[2,0]
   z_extent = ldef[2,1]

IDAOUT_inner:

    idaout_inner,fid,shotnum,iview,t,s,temperature,power,ldef,numsatpix,$
           alphaconst,tc_state,/scheduler,aug=aug, inner=inner

  
    iflag = iflag + 10

endif
;end of inner analysis

SKIP:
;
;  we always get here to close the ida file and put in the flag
;
idaout_close,fid,shotnum,iflag

end

