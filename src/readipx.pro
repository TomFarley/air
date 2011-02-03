function readipx,shot,framerate,integtime,numframes,width,height,xstart,ystart,gain,offset,data,filename,time,maxtime,lambda,targetrate=targetrate

print,'opening file',filename
   descr = ipx_open(filename)
;
;     to get the framerate need to read in the first two frames
;
   a = ipx_frame(descr,0,time=time0)
   a = ipx_frame(descr,1,time=time1)
   ;bg= ipx_frame(descr,1,/ref)
;
;   time comes back as double - part of the old code required time
;   as single precision therefore keep as single
;
   framerate = float(1./(time1-time0))



         sumframes=descr.header.numframes
	bunch=0
	if (shot ge 8801 and shot le 8823 ) then begin
		bunch=1 ;(FL: could this be replaced by a data file too?)
	endif
	if (shot ge 8889 and shot le 8909) then begin
		bunch=1
	endif

         integtime=descr.header.exposure *1e-6
	 numframes=0l
	if(bunch) then begin
		ysize_orig = descr.header.height
		ysize = 8
		numframes = descr.header.numframes*32
	endif else begin
		ysize_orig = descr.header.height
		numframes = descr.header.numframes
	endelse
         tmp=long(numframes)
	 numframes=tmp(0,0)
	
	 width=descr.header.width
         height=descr.header.height

         gain=descr.header.offset(0)
         offset=fix(descr.header.gain(0) )
         toffset =0 ;descr.fileinfo.trigger

;LWIR SETTINGS	
;	;images are stored as top right index (0,0) lower left index (320,255) 
;	 xstart=descr.header.left+descr.header.width-width
;         ystart=descr.header.top;yoffset in header is from top of image
 ;        ulrow=descr.header.left
  ;       ulcol=(descr.header.top+descr.header.height-1)-ystart
  ;;       lrrow=(descr.header.left+descr.header.width-1)-xstart
  ;       lrcol=(descr.header.top)-ystart

;MWIR SETTINGS	
	xstart=descr.header.right-descr.header.width
        ystart=descr.header.bottom-ysize_orig
        ulrow=descr.header.left-1
        ulcol=(descr.header.bottom-1)-ystart
        lrrow=(descr.header.right-1)-xstart
        lrcol=(descr.header.top-1)-ystart

;     get the information on which filter is used
;
          lambda = descr.header.filter

	 if keyword_set(targetrate) then begin ;calculate divisor for
	   dn=fix(0.5+framerate/targetrate)	; rate downsampling
	 endif
	 
; PRINT DATA IN HEADER ON SCREEN

        print,'********************* HEADER ****************************'
        print,'Date: ',descr.header.date_time
        print,'Size (X/Y)   : ',width,'/',height
        print,'Upper left corner (X/Y): ', ulrow,'/'  ,ulcol
        print,'Lower right corner (X/Y): ',lrrow,'/' ,lrcol
	print,'GrabberFrames  : ',numframes
	print,'f [Hz.] :',framerate;' not ',framerate
	print,'Int.(us):',integtime*1e6
        print,'Sum number frames: ',numframes
        print,'Revision: ',0                 
        print,'Gain, Offset: ',gain,offset
        print,'Xstart, Ystart: ',xstart,ystart
	;print, 'X offset, Y offset: ', xstart, descr.fileinfo.top-1>0
        print,'Maximal/Minimal Value of RawData :',0.,0.                          
; get proper ratio between frame size grapper and frame size picture

        ;dummy=getframevalues(framerate,width,height,numframes)

        print,'Derived Values -------------------------'
        print,'PictureFrames  : ',numframes
        print,'PictureSize    : ',width,height

; Change to numframes as given with maxtime
        nfnew=fix(maxtime*framerate)
        if (numframes ge nfnew) then begin 
        numframes=nfnew 
        print,'Numframes according to maxtime set to: ',numframes
        end

;  	READ IMAGES

   dataf=intarr(width,height)
   if not keyword_set(targetrate) then begin
	data=intarr(width,height,numframes)
bg=ipx_frame(descr,2,time=time0)	
;bg1=ipx_frame(descr,1,/ref)

;ref2=ipx_frame(descr,2,/ref)

;gain=(mean(ref2)-mean(bg1))/(ref2-bg1)

	for i=1,numframes-1  do begin
             dataf = (ipx_frame(descr,i,time=time0)-bg)
	                if (shot ge 18942 and shot le 18958) then dataf=reverse(dataf,1)
       
		data(*,*,i)=dataf

	endfor   
   endif else begin
   	data=fltarr(width,height,numframes/dn+1)
	numframes=0
	n=0
        newframes = 0
	
	for i=0,numframes-1  do begin
             dataf = (ipx_frame(descr,i,time=time0))
		if (shot ge 18942 and shot le 18958) then dataf = reverse(dataf,1)
	        if n mod dn eq 0 then begin
		  data(*,*,newframes)=dataf
		  newframes=newframes+1
		endif
             n=n+1
	endfor   
	data=data(*,*,0:newframes-1)
   endelse

        print,'Actual read-in Numframes: ',numframes
	if keyword_set(targetrate) then begin
	  framerate=framerate/dn
	  print,'Downsampled frame rate to',framerate
	endif

        data=float(data)

;       define time vector
        time=findgen(numframes)/framerate
        print,'Images read ...',time(0),time(numframes-1)

        print,'*********************************************************'
      
end
