;all prodedures for displaying things are in here
;originally in leon.pro

pro define_ccd,ptco,ro,phio,zo,alpha,beta,gamma,o,p,i,j,n,l,err

;ptco=0.0125d ;pinhole to CCD perpendicular distance in m

o=[ro*cos(phio/360.0*2*!pi),ro*sin(phio/360.0*2*!pi),zo]

;CCD is 320*256 and 30micron square pixels - for MWIR
;256x256 for LWIR.....
p=[[0,320,256],[0,-320,256],[0,-320,-256],[0,320,-256]]
p=p/2.0*30e-6

rot,p,gamma,/x
rot,p,beta,/y
rot,p,(phio+alpha),/z
trans,p,-o

i=p(*,1)-p(*,0)
i=i/sqrt(dot(i,i))
j=p(*,3)-p(*,0)
j=j/sqrt(dot(j,j))

norm,i,j,n

if(abs(phio+alpha) lt 90.0) then begin
	n=-n
endif else begin
	err='phio+alpha > 90.0 - help!'
	return
endelse

l=o+ptco*n

;show_ccd,o,n,p,alpha,beta,gamma,phio
wset,0

err=''

end

pro update_dist,ptc ;FL:This procedure added to update distortion common block from distort.dat
common distortion,dist

distarr=readtxtarray('distort.dat',2);FL:Reads in from 2-column data file, ptc and distortion
ptcrow=where(ptc gt distarr(0,*)-0.00052 and ptc lt distarr(0,*)+0.00052,noel)
;Some lea-way given on the lens being used
if noel ne 0 then begin
  dist=distarr(1,ptcrow)
  print,'Distortion factor set to',dist
endif else begin
  dist=0.0
  print,'Lens not recognised, distortion set to zero'
  ;stop
endelse;FL:Now, any new lenses only need to be added to distort.dat

end

pro image,x,o,n,i,j,l,r,imult,jmult,err
common distortion,dist
; determine vector for intersection of CCD plane and the 
; line between point in vessel and the lens

a=dot(n,(l-x))
if(a ne 0.0) then begin
	u=dot(n,(o-x))/a
endif else begin
	err='point to lens line does not cross CCD plane'
	return
endelse

r=x+u*(l-x)
if(u le 1.0) then begin
	err='point behind CCD' ;indicates point is behind CCD no in front'
	return
endif

; determine co-ordinates (in i,j) of image on CCD by vector analysis

ptc=(l(0)-o(0))/n(0)

q=x-o
if(total(abs(q-n)) eq 0.0) then begin
	err='point is at pinhole'
	return
endif

np=n*ptc
imult=(-dot(np,np)*dot(i,q))/(dot(q,np)-dot(np,np))
jmult=(-dot(np,np)*dot(j,q))/(dot(q,np)-dot(np,np))

rpred=sqrt(imult^2+jmult^2)

;dist=-1.0e5/36.0d*rpred^2;FL: This is GFC's original distortion, now common


;if ptc gt 0.01248 and ptc lt 0.01352 then $;is there a 13mm lens?
;  dist=-2370.*rpred*rpred$ ;value taken from Janos datasheet
;else if ptc gt 0.02448 and ptc lt 0.02552 then $ ;or is it a 25mm lens?
;  dist=-896.*rpred*rpred$  ;FL: "Could these values be calculated instead?"
;else dist=0.0             ;no distortion if it doesn't match a known lens

corr=(dist*rpred*rpred+1.0d)
if(corr lt 0.5) then corr=0.5
imult=imult*corr
jmult=jmult*corr

err=''

;test_vec,o,x,i,j,n,imult,jmult,ptc

end

pro ccd_image,o,p,i,j,n,l,ribs,boxs,p2,cc
;draws the wire frame over the image.

;FL's simplified ccd_image. Hopefully makes it easier to add new components.
drawvec,ribs,o,p,i,j,n,l
drawvec,boxs,o,p,i,j,n,l
drawvec,p2,o,p,i,j,n,l
drawvec,cc,o,p,i,j,n,l
end

pro drawvec,vec,o,p,i,j,n,l 
;FL:added this to tidy up ccd_image
for nr=0,n_elements(vec(0,0,0,*))-1 do begin
  for nv=0,n_elements(vec(0,0,*,0))-1 do begin
    q=vec(*,*,nv,nr)
    draw,q,o,i,j,n,l
  endfor
endfor
end

pro draw,q,o,i,j,n,l
p=q
for k1=0,n_elements(p(0,*))-1 do begin
k2=(k1+1) mod n_elements(p(0,*))

; interpolate between p(*,k2) and p(*,k1)

dx=(p(0,k2)-p(0,k1))/20.0
dy=(p(1,k2)-p(1,k1))/20.0
dz=(p(2,k2)-p(2,k1))/20.0
more=0
	for b=0,20 do begin
		x=[p(0,k1)+dx*b,p(1,k1)+dy*b,p(2,k1)+dz*b]
		image,x,o,n,i,j,l,r,imult,jmult,err
		if(err eq '') then begin
			if(more eq 0) then begin
 			plots,[imult],[jmult],col=1
				more=1
			endif else plots,[imult],[jmult],col=1,/continue
		endif
	endfor
endfor

end

pro test_vec,o,s,t,n,l,f,obj,img,gm,cm,dg,ddg,dm,ddm,hmin

k=(l(0)-o(0))/n(0)

v=s#replicate(1,n_elements(img(0,*)))+n#img(0,*)/k
w=t#replicate(1,n_elements(img(1,*)))+n#img(1,*)/k
q=f-o#replicate(1,n_elements(f(0,*)))

gm=[0.0,0.0,0.0]
cm=[0.0,0.0,0.0]

h=0.0
for i=0,n_elements(v(0,*))-1 do begin
	h=h+((dot(v(*,i),q(*,i))-img(0,i))^2+$
	(dot(w(*,i),q(*,i))-img(1,i))^2)*img(2,i)
endfor
hmin=h

for xg=-dg/2.0,dg/2.0,dg/ddg do begin

	rvx=v
	rwx=w
	for i=0,n_elements(v(0,*))-1 do begin
		rvxa=rvx(*,i)
		rwxa=rwx(*,i)
		rot,rvxa,xg,/x
		rot,rwxa,xg,/x	
		rvx(*,i)=rvxa
		rwx(*,i)=rwxa
	endfor

for yg=-dg/2.0,dg/2.0,dg/ddg do begin
	rvy=rvx
	rwy=rwx

	for i=0,n_elements(rvy(0,*))-1 do begin
		rvya=rvy(*,i)
		rwya=rwy(*,i)
		rot,rvya,yg,/y
		rot,rwya,yg,/y	
		rvy(*,i)=rvya
		rwy(*,i)=rwya
	endfor

for zg=-dg/2.0,dg/2.0,dg/ddg do begin
	rvz=rvy
	rwz=rwy
	for i=0,n_elements(rvz(0,*))-1 do begin
		rvza=rvz(*,i)
		rwza=rwz(*,i)
		rot,rvza,zg,/z
		rot,rwza,zg,/z	
		rvz(*,i)=rvza
		rwz(*,i)=rwza
	endfor

for xt=-dm/2.0,dm/2.0,dm/ddm do begin
for yt=-dm/2.0,dm/2.0,dm/ddm do begin
for zt=-dm/2.0,dm/2.0,dm/ddm do begin
	g=[xt,yt,zt]
	c=[xg,yg,zg]

	h=0.0
	for i=0,n_elements(rvz(0,*))-1 do begin
		h=h+(dot(rvz(*,i),(q(*,i)+g))-img(0,i))^2+$
		(dot(rwz(*,i),(q(*,i)+g))-img(1,i))^2
	endfor

	if(h lt hmin) then begin
		hmin=h
		gm=g
		cm=c
	endif
endfor
endfor
endfor
endfor
endfor
endfor

end

pro show_ccd,o,n,p,alpha,beta,gamma,phio
common invessel,ribs,boxs,p2,cc

z=[[0,320,256],[0,-320,256],[0,-320,-256],[0,320,-256]]
z=z/2.0*30e-6*100
z=p
trans,z,o
rot,z,-(phio+alpha),/z
rot,z,-beta,/y
rot,z,-gamma,/x
z=z*100
rot,z,gamma,/x
rot,z,beta,/y
rot,z,(phio+alpha),/z
trans,z,-o

m=o+0.2*n

window,2,xsize=400,ysize=400

!p.multi=[0,0,2]
plot,[100],[100],xr=[-3,3],yr=[-3,3],xs=5,ys=5
plots,[z(0,0),z(0,1)],[z(2,0),z(2,1)]
plots,[z(0,1),z(0,2)],[z(2,1),z(2,2)]
plots,[z(0,2),z(0,3)],[z(2,2),z(2,3)]
plots,[z(0,3),z(0,0)],[z(2,3),z(2,0)]
plots,[o(0),m(0)],[o(2),m(2)]

for nr=0,n_elements(ribs(0,0,0,*))-1 do begin
	for nv=0,n_elements(ribs(0,0,*,0))-1 do begin
		q=ribs(*,*,nv,nr)
		plots,[q(0,0),q(0,1)],[q(2,0),q(2,1)]
		plots,[q(0,0),q(0,1)],[q(2,0),q(2,1)],/continue
	endfor
endfor

for nr=0,n_elements(boxs(0,0,0,*))-1 do begin
	for nv=0,n_elements(boxs(0,0,*,0))-1 do begin
		q=boxs(*,*,nv,nr)
		plots,[q(0,0),q(0,1)],[q(2,0),q(2,1)]
		plots,[q(0,0),q(0,1)],[q(2,0),q(2,1)],/continue
	endfor
endfor

for nr=0,n_elements(p2(0,0,0,*))-1 do begin
	for nv=0,n_elements(p2(0,0,*,0))-1 do begin
		q=p2(*,*,nv,nr)
		plots,[q(0,0),q(0,1)],[q(2,0),q(2,1)]
		plots,[q(0,0),q(0,1)],[q(2,0),q(2,1)],/continue
	endfor
endfor

for nr=0,n_elements(cc(0,0,0,*))-1 do begin
	for nv=0,n_elements(cc(0,0,*,0))-1 do begin
		q=cc(*,*,nv,nr)
		plots,[q(0,0),q(0,1)],[q(2,0),q(2,1)]
		plots,[q(0,0),q(0,1)],[q(2,0),q(2,1)],/continue
	endfor
endfor

plot,[100],[100],xr=[-3,3],yr=[-3,3],xs=5,ys=5
plots,[z(0,0),z(0,1)],[z(1,0),z(1,1)]
plots,[z(0,1),z(0,2)],[z(1,1),z(1,2)]
plots,[z(0,2),z(0,3)],[z(1,2),z(1,3)]
plots,[z(0,3),z(0,0)],[z(1,3),z(1,0)]
plots,[o(0),m(0)],[o(1),m(1)]

for nr=0,n_elements(ribs(0,0,0,*))-1 do begin
	for nv=0,n_elements(ribs(0,0,*,0))-1 do begin
		q=ribs(*,*,nv,nr)
		plots,[q(0,0),q(0,1)],[q(1,0),q(1,1)]
		plots,[q(0,0),q(0,1)],[q(1,0),q(1,1)],/continue
	endfor
endfor

for nr=0,n_elements(boxs(0,0,0,*))-1 do begin
	for nv=0,n_elements(boxs(0,0,*,0))-1 do begin
		q=boxs(*,*,nv,nr)
		plots,[q(0,0),q(0,1)],[q(1,0),q(1,1)]
		plots,[q(0,0),q(0,1)],[q(1,0),q(1,1)],/continue
	endfor
endfor

for nr=0,n_elements(p2(0,0,0,*))-1 do begin
	for nv=0,n_elements(p2(0,0,*,0))-1 do begin
		q=p2(*,*,nv,nr)
		plots,[q(0,0),q(0,1)],[q(1,0),q(1,1)]
		plots,[q(0,0),q(0,1)],[q(1,0),q(1,1)],/continue
	endfor
endfor

for nr=0,n_elements(cc(0,0,0,*))-1 do begin
	for nv=0,n_elements(cc(0,0,*,0))-1 do begin
		q=cc(*,*,nv,nr)
		plots,[q(0,0),q(0,1)],[q(1,0),q(1,1)]
		plots,[q(0,0),q(0,1)],[q(1,0),q(1,1)],/continue
	endfor
endfor

!p.multi=0

end

pro point,f,o,p,i,j,n,l,imult,jmult,psy

image,f,o,n,i,j,l,r,imult,jmult,err

if(err eq '') then begin
	plots,[imult],[jmult],col=1,psym=psy
endif

end

pro undistort,xc,yc,xd,yd
common distortion, dist

xc=(xc-320)/2*30e-6
yc=(yc-256)/2*30e-6

rpred=sqrt(xc^2+yc^2)
;dist=1.0e5/36.0d*rpred^2 ;FL: This is GFC's original dist, now common 
corr=(-dist*rpred*rpred+1.0d)
xd=xc*corr
yd=yc*corr
end

pro delay

for y=0,100 do cursor,xc,yc,/nowait

end

pro display

;set up the scale on the images - upper and lower, based on the brightness of
;the image

set_display_at,/scon,w=0 ;image is TVed in here?

base1 = WIDGET_BASE(/column)
slide1=WIDGET_SLIDER(base1,title='lower',value=0,uvalue='lower')    
slide2=WIDGET_SLIDER(base1, title='upper',value=100,uvalue='upper')
button1=WIDGET_BUTTON(base1,value='Done',uvalue='done')
widget_control, base1, /realize
xmanager,'display',base1

end

pro display_event,ev

;event handler for setting of the upper and lower limits from display

WIDGET_CONTROL, ev.top
WIDGET_CONTROL, ev.id, GET_UVALUE=uval

case uval of
	'lower': set_display_at,b=ev.value,/scon
	'upper': set_display_at,c=ev.value,/scon
	'done': widget_control,ev.top,/destroy
endcase

end

pro frame

common toggle,tog,search
tog=1
search=0
set_frame

base3 = WIDGET_BASE(/column)
button1=WIDGET_BUTTON(base3,value='toggle frame',uvalue='toggle')    
button2=WIDGET_BUTTON(base3,value='new point',uvalue='new')
button3=WIDGET_BUTTON(base3,value='begin search',uvalue='search')
widget_control, base3, /realize
xmanager,'frame',base3

end

pro frame_event,ev
common toggle,tog,search

WIDGET_CONTROL, ev.top
WIDGET_CONTROL, ev.id, GET_UVALUE=uval

case uval of
	'toggle': begin
          set_display_at,w=tog
          tog=(tog+1) mod 2
          end
	'new': widget_control,ev.top,/destroy
	'search': begin
          search=1
          widget_control,ev.top,/destroy
        end
endcase

end

pro set_frame

common toggle,tog

set_display_at,w=tog
tog=(tog+1) mod 2

end

pro set_display,w=w,b=b,c=c,scon=scon

common data,d,width,height,xst,yst,comp,dnuc,pix,s
common image,bset,cset,bcut,tcut

e=d-dnuc ;perform NUC correction

if(keyword_set(scon)) then begin
	if(n_elements(b) ne 0) then bset=b ;use vals from slider if set
	if(n_elements(c) ne 0) then cset=c ;by display_event
	bcut=bset/100.0*(max(e)-min(e))+min(e) ;set vals if slider not defined
	tcut=cset/100.0*(max(e)-min(e))+min(e) ;for upper and lower limit

	if(bcut eq 0.0 and tcut eq 0.0) then begin
		print,'Contrast failure - using defaults values'
		bcut=0.0
		tcut=5.0
	endif
endif

;limit the range of the image based on the upper and lower limits set using
;display_event
q=where(e le bcut)
if(q(0) ne -1) then e(q)=bcut
q=where(e ge tcut)
if(q(0) ne -1) then e(q)=tcut
e=(e-bcut)/(tcut-bcut)

;scale the colours somehow
e=fix(255*e)
;flip around so that the image is the correct way around for the wireframe
e=transpose(reverse(transpose(e))) 
;make larger to fill window.
re=rebin(e,[n_elements(e(*,0))*2,n_elements(e(0,*))*2])

if(xst ne 0 or yst ne 0) then begin
	;print, 'Sub array'
;	print, xst, yst
	;Make ccd image same size as the wire frame image if offset >0
	;NOTE: x offset from left of image, y offset from top
	re_resized=make_array(640,512) ;define array
	re_resized[*,*]=255 ;set to white

	;plot sub array image onto full 640x512 image for overlay later
	;re_resized[left_edge:right_edge, bottom_edge:top_edge]
	;re_resized[xst-1:xst+width-1,yst-1:yst+height-1]=re
	;re=re_resized
	if n_elements(re_tmp) eq 0 then begin
		;print, 'Run plotting'
		loadct, 0, /silent
		tv, re_resized ;make plot of image 640x512
		tv, re, xst, yst ;place sub array on 640x512 image
		re_tmp=tvrd()  ;store 640x512 image in variable for use later
		re=re_tmp
	endif
endif

;when set_display is called with w ne 0, then the plot is for the image
; overlayed with the wireframe
if(n_elements(w) ne 0) then begin

	if(w eq 1) then begin
		;wireframe (in variable comp) is a black on white image. 
		;turn it round so that where it is combined with the IR image
		;wireframe is white
		q=where(comp eq 255)
		r=where(comp eq 0)
		comp[q]=0
		comp[r]=255
		re[q]=255 ;make the parts of the image where the wireframe is white
	endif
endif

;stop

;converting image to 3d array. Related to decomposed=0?
ns = size(re)
re3 = intarr(3,ns[1],ns[2])
for k=0,2 do re3(k,*,*) = re
tv,re3, /true;,xst,yst,/true
end

pro av_image,time,data,tav,dav,framerate

dt=1.0/framerate
d=fltarr(n_elements(data(*,0)),n_elements(data(0,*)))
sav=0.0
i=0
for s=tav(0),tav(1),dt do begin
	i=i+1
	q=where(abs(time-s) eq min(abs(time-s)))
	sav=sav+time(q(0))
;print,'Processing frame for time',time(q(0))
	d=d+data(*,*,q(0))
endfor
tav=sav/float(i)
dav=d/float(i)

end

