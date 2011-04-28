;alignment procedures
;previously in leon.pro

pro vectors,ptco,ro,phio,zo,alpha,beta,gamma

;This program defines the position of the vectors and allows them
;to be overplotted onto the image from the camera

common vectors,ptc1,r1,phi1,z1,a1,b1,g1
;typical lower position
;ptc0=0.0125d ;pinhole to ccd length (focal length) in m
;ro=2.3d ;radial position, greater than zero
;phio=12.416 ;toroidal angle +/- 180.0
;zo=-1.195d
;alpha=-20.0d ;horizontal angle wrt to radius (0=radial, inverted) - 'yaw'
;beta=25d ;angle below horizontal (-89.999 to 89.999) - 'pitch'
;gamma=0.0d ;rotational angle wrt radius (0=horizontal) - 'roll'
;
;typical upper position
;ptc0=0.0125d ;pinhole to ccd length (focal length) in m
;ro=2.1d ;radial position, greater than zero
;phio=12.416 ;toroidal angle +/- 180.0
;zo=1.195d
;alpha=-30.0d ;horizontal angle wrt to radius (0=radial, inverted) - 'yaw'
;beta=-30d ;angle below horizontal (-89.999 to 89.999) - 'pitch'
;gamma=35.0d ;rotational angle wrt radius (0=horizontal) - 'roll'

;ptc1=0.0150 ;FL: changed from 0.0125
;update_dist,ptc1                ;FL: updating distortion common block
;r1=2.20
;phi1=5.2646
;z1=-1.301
;a1=-17.3659
;b1=26.7525
;g1=-41.6
;
; alignement
;
;view #1 upper
ptc1=0.0150 ;FL: changed from 0.0125
update_dist,ptc1                ;FL: updating distortion common block
r1=2.18
phi1=-27.24
z1=1.3
a1=0.
b1=-18.6
g1=-88.

;view #1 lower
;ptc1=0.0130 ;FL: changed from 0.0125
;update_dist,ptc1                ;FL: updating distortion common block
;r1=2.207
;phi1=-38.
;z1=-1.27
;a1=0.8
;b1=17.81
;g1=87.




set_vectors,/scon

base2=WIDGET_BASE(/column)
field1=CW_FIELD(base2,title='focal length (m)',value=ptc1,uvalue='ptc',/RETURN_EVENTS)
field2=CW_FIELD(base2,title='radius (m)',value=r1,uvalue='r',/RETURN_EVENTS)
field3=CW_FIELD(base2,title='toroidal angle (deg)',value=phi1,uvalue='phi',/RETURN_EVENTS)
field4=CW_FIELD(base2,title='height (m)',value=z1,uvalue='z',/RETURN_EVENTS)
field5=CW_FIELD(base2,title='yaw',value=a1,uvalue='yaw',/RETURN_EVENTS)
field6=CW_FIELD(base2,title='pitch',value=b1,uvalue='pitch',/RETURN_EVENTS)
field7=CW_FIELD(base2,title='roll',value=g1,uvalue='roll',/RETURN_EVENTS)
button1=WIDGET_BUTTON(base2,value='Done',uvalue='done')
widget_control, base2, /realize
xmanager,'vectors',base2

ptco=ptc1
ro=r1
phio=phi1
zo=z1
alpha=a1
beta=b1
gamma=g1

end

pro vectors_event,ev

WIDGET_CONTROL, ev.top
WIDGET_CONTROL, ev.id, GET_UVALUE=uval

case uval of
	'ptc': set_vectors,ptcx=ev.value,/scon
	'r': set_vectors,rx=ev.value,/scon
	'phi': set_vectors,px=ev.value,/scon
	'z': set_vectors,zx=ev.value,/scon
	'yaw': set_vectors,ax=ev.value,/scon
	'pitch': set_vectors,bx=ev.value,/scon
	'roll': set_vectors,gx=ev.value,/scon
	'done': widget_control,ev.top,/destroy
endcase

end

pro set_vectors,ptcx=ptcx,rx=rx,px=px,zx=zx,ax=ax,bx=bx,gx=gx,scon=scon
common vectors,ptc1,r1,phi1,z1,a1,b1,g1
common invessel,ribs,boxs,p2,cc

if(n_elements(ptcx) ne 0) then begin
  ptc1=(ptcx*1.0)[0]
  update_dist,ptc1              ;FL: This included to update distortion
endif
if(n_elements(rx) ne 0) then r1=rx*1.0
if(n_elements(px) ne 0) then phi1=px*1.0
if(n_elements(zx) ne 0) then z1=zx*1.0
if(n_elements(ax) ne 0) then a1=ax*1.0
if(n_elements(bx) ne 0) then b1=bx*1.0
if(n_elements(gx) ne 0) then g1=gx*1.0

define_ccd,ptc1,r1,phi1,z1,a1,b1,g1,o,p,i,j,n,l,err

if(err ne '') then return

!p.multi=0
plot,[100],[100],xr=[-320,320]/2*30e-6,yr=[-256,256]/2*30e-6,xs=5,ys=5,/noerase
erase
set_display_at,w=0
;print,'in vectors ',ptc1,r1,phi1,z1,a1,b1,g1
ccd_image_at,o,p,i,j,n,l,ribs,boxs,p2,cc

end

pro specify_point,o,p,i,j,n,l,ribs,boxs,p2,cc,img,obj,f
common toggle,tog,search
common data,d,width,height,xst,yst,comp,dnuc,pix,s

a=findgen(17)*(!pi*2/16.0)
usersym,cos(a),sin(a)

erase
display

while(1) do begin
  xdo=-1.0
  ydo=-1.0
  xd=1.0
  yd=1.0
  set_display_at,w=0
  frame
  if(search) then begin
    if(n_elements(f(0,*)) eq 1) then begin
      f=0
    endif else begin
      img=img(*,1:*)
      obj=obj(*,1:*)
      f=f(*,1:*)
    endelse
    return
  endif

  while (xdo ne xd or ydo ne yd) do begin
    xdo=xd
    ydo=yd
    erase
    set_display_at,w=1
    plots,xd,yd,col=1,psym=8
    cursor,xc,yc,/device,/up
    delay
    undistort,xc,yc
    
    find_bestvec,xc,yc,o,p,i,j,n,l,ribs,boxs,p2,cc,bestvec
    image,bestvec,o,n,i,j,l,r,xd,yd,err
    plots,xd,yd,col=truecolor('royalblue'),psym=8
  endwhile

  obj=[[obj],[xd,yd]]
  f=[[f],[bestvec]]
  xdo=-1.0
  ydo=-1.0
  xd=1.0
  yd=1.0
  xc=1.0
  yc=1.0

  while (xdo ne xd or ydo ne yd) do begin
    xdo=xd
    ydo=yd
    erase
    set_display_at,w=0
    plots,xc,yc,col=1,psym=8
    cursor,xc,yc,/device,/up
    delay
    undistort,xc,yc,xd,yd
    plots,xc,yc,col=1,psym=8
  endwhile

  img=[[img],[xd,yd,1]]
  usersym,cos(a),sin(a),/fill
  plots,xc,yc,col=1,psym=8
endwhile
end

;************** GFC ******************
pro find_vectors,ptco,o,p,i,j,n,l,ribs,boxs,p2,cc,f,obj,img,close,new=new,disp=disp

common data,d,width,height,xst,yst,comp,dnuc,pix,s
common findvec,gm,cm,dg,ddg,dm,ddm,hold

if(keyword_set(new)) then begin
  hold=1.0e38
  dg=16.0
  ddg=8.0
  dm=0.16
  ddm=8.0
endif

top:
erase
ccd_image_at,o,p,i,j,n,l,ribs,boxs,p2,cc
if (keyword_set(disp)) then comp=tvrd()
set_display_at,w=1

test_vec,o,i,j,n,l,f,obj,img,gm,cm,dg,ddg,dm,ddm,hmin

no=n
io=i
jo=j
rot,n,cm(0),/x
rot,n,cm(1),/y
rot,n,cm(2),/z
rot,i,cm(0),/x
rot,i,cm(1),/y
rot,i,cm(2),/z
rot,j,cm(0),/x
rot,j,cm(1),/y
rot,j,cm(2),/z
o=o-gm
;************** GFC ******************
l=o+ptco*n

ro=sqrt(o(0)^2+o(1)^2)
print,'ro=',ro

phio=atan(o(1),o(0))/2.0/!pi*360.0
print,'phio=',phio

zo=o(2)
print,'zo=',zo

b=(-n*[1,1,0])/sqrt(dot((-n*[1,1,0]),(-n*[1,1,0])))  

alpha=-acos(dot(b,[1,0,0]))/2.0/!pi*360.0-phio
print,'alpha=',alpha

beta=acos(-dot(-n,[0,0,1]))/2.0/!pi*360.0-90.0
print,'beta=',beta

z=j
rot,z,-(phio+alpha),/z
rot,z,-beta,/y
gamma=-acos(dot(-z,[0,0,1]))/2/!pi*360.0
print,'gamma=',gamma

dg=dg/2.0
dm=dm/2.0

hdiff=(hold-hmin)/hold
print,hold,hmin,hdiff
hold=hmin

if(hdiff gt close) then goto,top

spawn,'whoami',mename
save,filename=mename+'_vectors.dat',o,p,i,j,n,l
close=hdiff

end

pro find_bestvec,xd,yd,o,p,i,j,n,l,ribs,boxs,p2,cc,bestvec

rms=1.0e38
for ncom=0,n_elements(ribs(0,0,0,*))-1 do begin
    for nvec=0,n_elements(ribs(0,0,*,0))-1 do begin
        image,ribs(*,0,nvec,ncom),o,n,i,j,l,r,imult,jmult,err
        if(err eq '') then begin
            dist=total((xd-imult)^2+(yd-jmult)^2)
            if(dist lt rms) then begin
                rms=dist
                bestvec=ribs(*,0,nvec,ncom)
            endif
        endif
    endfor
endfor

for ncom=0,n_elements(boxs(0,0,0,*))-1 do begin
    for nvec=0,n_elements(boxs(0,0,*,0))-1 do begin
        image,boxs(*,0,nvec,ncom),o,n,i,j,l,r,imult,jmult,err
        if(err eq '') then begin
            dist=total((xd-imult)^2+(yd-jmult)^2)
            if(dist lt rms) then begin
                rms=dist
                bestvec=boxs(*,0,nvec,ncom)
            endif
        endif
    endfor
endfor

end
