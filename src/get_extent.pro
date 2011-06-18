FUNCTION get_extent,theo,o,n,i,j,l,r

;program to determine the extent of the analysis line across the
;CCD image

;the purpose is to determine the number of pixels crossed and 
;hence the maximum number of steps along the analysis path
;which can be used.

;For now, simply run the code with a large number of steps
;and find the range over which the data is to be extracted.
;Feed the range back into the get_line program and repeat
;with the correct number of steps.

common data,d,width,height,xst,yst,comp,dnuc,pix,s

steps=200

cx=[-320,320]/2*30e-6
cy=[-256,256]/2*30e-6

start=rtptoxyz(theo(*,0))
finish=rtptoxyz(theo(*,0)+theo(*,1))

p=[[start],[finish]]

; interpolate between p(*,k2) and p(*,k1)
dx=(p(0,1)-p(0,0))/float(steps)
dy=(p(1,1)-p(1,0))/float(steps)
dz=(p(2,1)-p(2,0))/float(steps)

muli=[0.0]
mulj=[0.0]
sl=[0.0]

for b=0,steps do begin
  x=[p(0,0)+dx*b,p(1,0)+dy*b,p(2,0)+dz*b]
  image,x,o,n,i,j,l,r,imult,jmult,err
  if(err eq '') then begin
    muli=[muli,imult]
    mulj=[mulj,jmult]
    sl=[sl,sqrt(total((x-start)^2))]
  endif
endfor
mul=[[muli(1:*)],[mulj(1:*)]]
sl=sl(1:*)
pixl=[[fix((mul(*,0)-cx(0))/(cx(1)-cx(0))*319)],$
     [fix((mul(*,1)-cy(0))/(cy(1)-cy(0))*255)]]

; must be a better way to do this!

;determine the location of the pixels to extract from the camera data
;which correspond to the anaylsis path
pix2=[0,0]
s2=[0.0]
for ik=0,n_elements(pixl(*,0))-1 do begin
  apix=(pixl(ik,0)-xst/2)
  bpix=(pixl(ik,1)-yst/2)
  if(apix ge 0 and bpix ge 0 and apix lt width and bpix lt height) then begin
    pix2=[[pix2],[apix,bpix]]
    s2=[s2,sl(ik)]
  endif
endfor

;the analysis path may lie in the y direction
;or the x direction - find the largest
;is this the best way - does the largest direction
;always go in the direction of increasing radius?

;first element of pix2 is zero - remove this
pix2_cut=pix2[*,1:n_elements(pix2[0,*])-1]
range_x_direction=max(pix2_cut[0,*])-min(pix2_cut[0,*])
range_y_direction=max(pix2_cut[1,*])-min(pix2_cut[1,*])

step=[range_x_direction,range_y_direction]

line_segments=max(step)

;line segments is the number of pixels which overlap with
;the ccd image - However, the analysis line can by longer
;than this. Need to determine how many steps are required
;to give steps=line_segments aross the WHOLE analysis line


;ATHORN (16/06/11) first go - end up with too many elements...
;so remove and try another method...

	;determine the delta_r over which there is overlap of the
	;number of pixels given by line_segments
	;delta_r=s2[n_elements(s2)-1]-s2[1]
	;compute the radial resolution of the overlapping line
	;and image
	;resolution_overlapping=delta_r/line_segments

	;compute the length of the WHOLE analysis line
	;delta_whole_line=sl[n_elements(sl)-1]-sl[1]
	
	;determine the number of steps along the WHOLE line to
	;match the radial resolution of the overlapped region
	;final_segments=delta_whole_line/resolution_overlapping

;ATHORN (17/06/11) second (simpler) go

;determine the ratio of the number of elements of sl to s2
ratio=float(n_elements(sl))/float(n_elements(s2))

;scale the number of segments the line is to be split into
;by the ratio defined above.
final_segments=ratio*float(line_segments)
;final_segments=line_segments

;print, final_segments
;stop
return, final_segments

END
