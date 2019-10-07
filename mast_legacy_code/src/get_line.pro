pro get_line,theo,o,n,i,j,l,r,steps,sl,pixl,disp=disp

common data,d,width,height,xst,yst,comp,dnuc,pix,s

;Define the number of segments the analysis path is to
;be split into - needs to be integer (as loop 
;variable later)

;if steps is defined in here then the number of steps will
;vary when multiple lines are run when the analysis path
;is an area - this causes the analysis code to produce
;too many position steps - so move the determination of
;step to get_trace.
if undefined(steps) eq 1 then begin	
	step=100
endif else begin
	step=fix(steps)
endelse
;step=fix(get_extent(theo,o,n,i,j,l,r))

cx=[-320,320]/2*30e-6
cy=[-256,256]/2*30e-6

start=rtptoxyz(theo(*,0))
finish=rtptoxyz(theo(*,0)+theo(*,1))

p=[[start],[finish]]

; interpolate between p(*,k2) and p(*,k1)
dx=(p(0,1)-p(0,0))/float(step)
dy=(p(1,1)-p(1,0))/float(step)
dz=(p(2,1)-p(2,0))/float(step)

muli=[0.0]
mulj=[0.0]
sl=[0.0]

for b=0,step do begin
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

;pix2 = location in CCD image of the pixels to extract (x,y)
;s2 = real radius of the pixels defined in s2.

;stop

;   whatever happens we want to see the line  mod by ak on 7/12/06
  if(keyword_set(disp)) then begin
    wset,0
    plots,mul(*,0),mul(*,1),col=truecolor('firebrick')
  endif

if(n_elements(s2) gt 1) then begin
  pixl=transpose(pix2(*,1:*))
  sl=s2(1:*)

	;ATHORN 17/06/11 - remove duplicated pixels
	unique_pixels=uniq(pixl[*,0])
	pixl=pixl[unique_pixels,*]
	sl=sl[unique_pixels]
	;ATHORN mods end.
;stop
  if(keyword_set(disp)) then begin
    wset,0
    plots,mul(*,0),mul(*,1),col=truecolor('firebrick')
  endif
endif else begin
  pixl=[0,0]
  sl=[0.0]
endelse
end
