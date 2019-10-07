
pro components,sh,ribsplus,tiles,bdump,p2plus,cc,pf=pf,gdc=gdc,mid=mid
;FL 10/8/2004 adding toggles for including GDC&Beamdumps, and PF coils
;as well as shot number input so geometry can be different post shot 10000
;FL 16/8/2004 adding logic on shot number to toggle new geometry
;GdT 9/12/08 added DSF (only 12 vectors out of 20 but enough to zwork with and makes life easier)
if sh lt 10000 and not keyword_set(mid) then begin
  get_ribs,ribs
  get_boxs,tiles
  get_p2,p2
  get_cc,cc
endif else begin
  get_louvres,tiles
  get_p2mid,p2
  get_cc,cc,/mid
  ribs=0
  endelse
if keyword_set(pf) then begin ;Check whether we need coils included
  get_pf,p3,p4,p5,p6
;FL: OK then here's a trick.  Let's graft the PF vectors onto the P2 vector.
;This way Leon needn't be changed
  p2plus=fltarr(3,2,36,44,/nozero)
  p2plus[*,*,*,0:11]=p2
  p2plus[*,*,*,12:19]=p3
  p2plus[*,*,*,20:27]=p4
  p2plus[*,*,*,28:35]=p5
  p2plus[*,*,*,36:43]=p6
endif else p2plus=p2
if keyword_set(gdc) then begin
  get_bdump,sh,bdump ;FL: Need to make this compatible with Leon (was 3,2,12,5)
  bd0=n_elements(ribs)/3/2/30;Check we've used ribs
  ribsplus=fltarr(3,2,30,bd0+11)    ;so let's add it to to the ribs array
  if bd0 gt 0 then ribsplus[*,*,*,0:bd0-1]=ribs
  ribsplus[*,*,0:11,bd0:bd0+5]=bdump[*,*,*,0:5]
  ribsplus[*,*,12:23,bd0+6:bd0+10]=bdump[*,*,*,6:10]
endif else ribsplus=ribs;At the moment new geometry needs GDC etc to work
end

function cuboid,r,phi,z,dr,dphid,dz,thetad
;FL:Defines a cuboid, originally for use in beam dump
;theta is the angle of rotation of the cuboid
phi1=-(phi-dphid/2)/180.0*!pi
phi2=-(phi+dphid/2)/180.0*!pi
theta=-thetad/180.0*!pi
dphi=-dphid/180.0*!pi

costheta=cos(theta);/180.0*!pi)
costpdp=cos(theta+dphi/2);/180.0*!pi)
costmdp=cos(theta-dphi/2);/180.0*!pi)
r1L=r*costheta/costmdp
r1R=r*costheta/costpdp
r2L=sqrt(r1L*r1L+dr*dr+2*r1L*dr*costmdp)
r2R=sqrt(r1R*r1R+dr*dr+2*r1R*dr*costpdp)

phi1b=(phi1-asin(dr/r2L*sin(theta-dphi/2)))
phi2b=(phi2-asin(dr/r2R*sin(theta+dphi/2)))

z1=z-dz/2
z2=z+dz/2

;cuboid=[$;This all turned out to be wrong
;       [[r1L,phi1,z1],[r2L,phi1b,z1]], [[r2L,phi1b,z1],[r2R,phi2b,z1]],$;\ Top
;       [[r2R,phi2b,z1],[r1R,phi2,z1]], [[r1R,phi2,z1],[r1L,phi1,z1]],$ ; /rect
;       [[r1L,phi1,z2],[r2L,phi1b,z2]], [[r2L,phi1b,z2],[r2R,phi2b,z2]],$;\ Btm
;       [[r2R,phi2b,z2],[r1R,phi2,z2]], [[r1R,phi2,z2],[r1L,phi1,z2]],$;  /rect
;       [[r1L,phi1,z1],[r1L,phi1,z2]], [[r2L,phi1b,z1],[r2L,phi1b,z2]],$; \Side
;       [[r2R,phi2b,z1],[r2R,phi2b,z2]], [[r1R,phi2,z1],[r1R,phi2,z2]] $; /corn
;       ]
xy=[[r1L*cos(phi1),r1L*sin(phi1)], [r2L*cos(phi1b),r2L*sin(phi1b)],$
    [r2R*cos(phi2b),r2R*sin(phi2b)], [r1R*cos(phi2),r1R*sin(phi2)]]
cuboid=[$
       [[xy(*,0),z1],[xy(*,1),z1]], [[xy(*,1),z1],[xy(*,2),z1]],$;\ Top
       [[xy(*,2),z1],[xy(*,3),z1]], [[xy(*,3),z1],[xy(*,0),z1]],$ ; /rect
       [[xy(*,0),z2],[xy(*,1),z2]], [[xy(*,1),z2],[xy(*,2),z2]],$;\ Btm
       [[xy(*,2),z2],[xy(*,3),z2]], [[xy(*,3),z2],[xy(*,0),z2]],$;  /rect
       [[xy(*,0),z1],[xy(*,0),z2]], [[xy(*,1),z1],[xy(*,1),z2]],$; \ Side
       [[xy(*,2),z1],[xy(*,2),z2]], [[xy(*,3),z1],[xy(*,3),z2]] $; /corns
       ]
return,cuboid
end

pro get_bdump,sh,bdump
;FL: Defines beamdumps using cuboid function, include GDC
bdump=fltarr(3,2,12,12,/nozero)
bdump[*,*,*,0]=cuboid(1.707,15.6,0.0,0.03,2.2,0.300,60.6);Angled tile
bdump[*,*,*,1]=cuboid(1.837,26.8,0.3,0.03,25.6,0.300,11.8);top row of beamdump
bdump[0:1,*,*,2]=bdump[0:1,*,*,1];copying to bottom row of beamdump
bdump[2,*,*,2]=-bdump[2,*,*,1]  ;flip z coord for bottom row
bdump[*,*,*,3]=cuboid(1.837,26.8,0.0,0.03,20.1,0.300,11.8);middle row
bdump[*,*,*,4]=cuboid(1.837,60,0.0,0.02,8.6,0.325,0);GDC at 60 degrees
bdump[*,*,*,5]=cuboid(1.837,-60,0.0,0.02,8.6,0.325,0);GDC at -60 degrees

iphioff = 0.
if (sh ge 18942 and sh le 18958) then iphioff = 10.
bdump[*,*,*,6]=cuboid(1.707,75.6+iphioff,0.0,0.03,2.2,0.300,60.6);Angled tile
bdump[*,*,*,7]=cuboid(1.837,86.8+iphioff,0.3,0.03,25.6,0.300,11.8);top row of beamdump
bdump[0:1,*,*,8]=bdump[0:1,*,*,7];copying to bottom row of beamdump
bdump[2,*,*,8]=-bdump[2,*,*,7]  ;flip z coord for bottom row
bdump[*,*,*,9]=cuboid(1.837,86.8+iphioff,0.0,0.03,20.1,0.300,11.8);middle row
bdump[*,*,*,10]=cuboid(1.837,-150,0.0,0.02,8.6,0.325,0);GDC at -150 degrees
bdump[*,*,*,11]=[$	;DSF
[[0.953,-.156,-1.827],[0.992,-0.164,-1.827]],$
[[0.992,-0.164,-1.827],[0.985,-0.199,-1.829]],$
[[0.985,-0.199,-1.829],[0.947,-0.192,-1.829]],$
[[0.947,-0.192,-1.829],[0.953,-0.156,-1.827]],$

[[0.953,-.156,-1.827],[0.992,-0.164,-1.827]],$
[[0.992,-0.164,-1.827],[0.992,-0.164,-1.865]],$
[[0.992,-0.164,-1.865],[0.992,-0.164,-1.827]],$
[[0.992,-0.164,-1.827],[0.953,-0.156,-1.827]],$


[[0.992,-.164,-1.827],[0.985,-0.199,-1.829]],$
[[0.985,-0.199,-1.829],[0.985,-0.199,-1.865]],$
[[0.985,-0.199,-1.865],[0.992,-0.164,-1.865]],$
[[0.992,-0.164,-1.865],[0.992,-0.164,-1.827]]$
]

;bdump[*,*,*,6:9]=bdump[*,*,*,0:3];copy first beam dump
;rot,bdump[*,*,*,6:9],60,/z      ;and rotate to second position
;for n=6,9 do for m=0,11 do rot,bdump[*,*,m,n],60,/z
end

pro get_ribs,ribs

;DEFINE RIBS

;define ribs vector for 22 ribs composed of 30 vectors each
ribs=fltarr(3,2,30,22)

;define a single lower rib
q=[$
[[0.7,0.025,-2.152],[0.7,0.025,-1.991]],$
[[0.7,0.025,-1.991],[1.65,0.025,-1.991]],$
[[1.65,0.025,-1.991],[1.65,0.025,-2.152]],$
[[1.65,0.025,-2.152],[0.7,0.025,-2.152]],$

[[0.7,0.033,-1.981],[1.65,0.033,-1.981]],$
[[1.65,0.033,-1.981],[1.65,0.057,-1.981]],$
[[1.65,0.057,-1.981],[0.7,0.057,-1.981]],$
[[0.7,0.057,-1.981],[0.7,0.033,-1.981]],$

[[0.7,0.065,-2.152],[0.7,0.065,-1.991]],$
[[0.7,0.065,-1.991],[1.65,0.065,-1.991]],$
[[1.65,0.065,-1.991],[1.65,0.065,-2.152]],$
[[1.65,0.065,-2.152],[0.7,0.065,-2.152]],$

[[0.7,0.025,-2.152],[0.7,0.065,-2.152]],$
[[1.65,0.025,-2.152],[1.65,0.065,-2.152]],$
[[0.7,0.025,-1.991],[0.7,0.033,-1.981]],$
[[1.65,0.025,-1.991],[1.65,0.033,-1.981]],$
[[0.7,0.057,-1.981],[0.7,0.065,-1.991]],$
[[1.65,0.057,-1.981],[1.65,0.065,-1.991]],$

[[1.017,0.033,-1.981],[1.017,0.057,-1.981]],$
[[1.017,0.057,-1.981],[1.017,0.065,-1.991]],$
[[1.017,0.065,-1.991],[1.017,0.065,-2.152]],$
[[1.017,0.065,-2.152],[1.017,0.025,-2.152]],$
[[1.017,0.025,-2.152],[1.017,0.025,-1.991]],$
[[1.017,0.025,-1.991],[1.017,0.033,-1.981]],$

[[1.333,0.033,-1.981],[1.333,0.057,-1.981]],$
[[1.333,0.057,-1.981],[1.333,0.065,-1.991]],$
[[1.333,0.065,-1.991],[1.333,0.065,-2.152]],$
[[1.333,0.065,-2.152],[1.333,0.025,-2.152]],$
[[1.333,0.025,-2.152],[1.333,0.025,-1.991]],$
[[1.333,0.025,-1.991],[1.333,0.033,-1.981]]$
]

;rotate by 30deg for full 2*pi - skip location of box ribs (60deg and 180deg on lower)
ia=0
for ang=0.0,360.0,30.0 do begin
if(ang eq 60.0 or ang eq 180.0) then goto,jump1
	qrot=q*0.0
	for i=0,n_elements(q(0,0,*))-1 do begin
		p=q(*,*,i)
		rot,p,ang,/z
		qrot(*,*,i)=p
	endfor
ribs(*,*,*,ia)=qrot
ia=ia+1
jump1:
endfor

;define a single upper rib
q(2,*,*)=-q(2,*,*)
q(1,*,*)=-q(1,*,*)

;rotate by 30deg for full 2*pi - skip location of box ribs (300deg and 60deg on upper)
ia=0
for ang=0.0,360.0,30.0 do begin
if(ang eq 300.0 or ang eq 180.0) then goto,jump2
	qrot=q*0.0
	for i=0,n_elements(q(0,0,*))-1 do begin
		p=q(*,*,i)
		rot,p,ang,/z
		qrot(*,*,i)=p
	endfor
ribs(*,*,*,11+ia)=qrot
ia=ia+1
jump2:
endfor

end


pro get_louvres,louvres;This should replace boxs for MID geometry post 10000

louvres=fltarr(3,2,20,96)
;Get corner coordinates for the first louvre in sector 1, then copy
;IN X,Y,Z COORDS
zT=-1.825 ;height of high top edge of louvre (low  top edge is variable)
zB=-1.865 ;height of base
;leftxy=fltarr(2,4,/nozero) ;xy coords of left and right edges of tiles.
;rightxy=fltarr(2,4,/nozero);Arranged start tile 1, end tile 1, end2, end3
leftxy= [[0.00,0.757],[0.008,1.053],[0.011,1.349],[0.017,1.698]]
rightxy=[[0.090,0.753],[0.125,1.046],[0.160,1.341],[0.199,1.688]]
;The following are to calculate the height of the tile after a 4 deg slope
leftzT=sqrt((leftxy(0,*)-rightxy(0,*))*(leftxy(0,*)-rightxy(0,*))$
            +(leftxy(1,*)-rightxy(1,*))*(leftxy(1,*)-rightxy(1,*)))
leftzT=zT-leftzT*tan(4*!pi/180.)

;Now make vector of first tile, starting with radial edges
louvre1=fltarr(3,2,20)
louvre1(*,*,0:3)=[[[leftxy(*,0),leftzT(0)],[leftxy(*,3),leftzT(3)]],$;top left
                  [[rightxy(*,0),zT],[rightxy(*,3),zT]],$ ;top right
                  [[leftxy(*,0),zB],[leftxy(*,3),zB]],$ ;bottom left
                  [[rightxy(*,0),zB],[rightxy(*,3),zB]]] ;bottom right
for i=0,3 do begin
;now tile ends
  louvre1(*,*,4*(i+1):4*i+7)=[[[leftxy(*,i),leftzT(i)],[leftxy(*,i),zB]],$
                             [[leftxy(*,i),zB],[rightxy(*,i),zB]],$
                             [[rightxy(*,i),zB],[rightxy(*,i),zT]],$
                             [[rightxy(*,i),zT],[leftxy(*,i),leftzT(i)]]]
endfor

;Now rotate all the louvres into place
louvres(*,*,*,0)=louvre1
for n=1,47 do begin
  ;first loop through all elements of louvre1 are rotate
  for i=0,n_elements(louvre1(0,0,*))-1 do begin
    edge=louvre1(*,*,i)
    rot,edge,7.5,/z;Can't just act on louvre1, unfortunately
    louvre1(*,*,i)=edge
  endfor
  louvres(*,*,*,n)=louvre1
endfor
;Now we need to make the top louvres, so rotate 180 deg about the x axis
louvres(*,*,*,48:95)=louvres(*,*,*,0:47)
for n=48,95 do for i=0,n_elements(louvre1(0,0,*))-1 do begin
  edge=louvres(*,*,i,n-48)
  rot,edge,180,/x
  louvres(*,*,i,n)=edge
endfor
dsf=[$
[[0.953,-.156,-1.827],[0.992,-0.164,-1.827]],$
[[0.992,-0.164,-1.827],[0.985,-0.199,-1.829]],$
[[0.985,-0.199,-1.829],[0.947,-0.192,-1.829]],$
[[0.947,-0.192,-1.829],[0.953,-0.156,-1.827]]$
]
end

pro get_boxs,boxs

;DEFINE BOXS

;define ribs vector for 4 ribs composed of 35 vectors each
boxs=fltarr(3,2,35,4)

;define a single lower box
q=[$
[[0.7252,-0.0959,-2.170],[0.7252,-0.0959,-1.99]],$
[[0.7252,-0.0959,-1.99],[1.6529,-0.0959,-1.99]],$
[[1.6529,-0.0959,-1.99],[1.6529,-0.0959,-2.170]],$
[[1.6529,-0.0959,-2.170],[0.7252,-0.0959,-2.170]],$

[[0.7252,-0.0879,-1.982],[1.6529,-0.0879,-1.982]],$
[[1.6529,-0.0879,-1.982],[1.6529,0.0879,-1.982]],$
[[1.6529,0.0879,-1.982],[0.7252,0.0879,-1.982]],$
[[0.7252,0.0879,-1.982],[0.7252,-0.0879,-1.982]],$

[[0.7252,0.0959,-2.170],[0.7252,0.0959,-1.99]],$
[[0.7252,0.0959,-1.99],[1.6529,0.0959,-1.99]],$
[[1.6529,0.0959,-1.99],[1.6529,0.0959,-2.170]],$
[[1.6529,0.0959,-2.170],[0.7252,0.0959,-2.170]],$

[[0.7252,-0.0959,-2.170],[0.7252,0.0959,-2.170]],$
[[1.6529,-0.0959,-2.170],[1.6529,0.0959,-2.170]],$
[[0.7252,-0.0959,-1.99],[0.7252,-0.0879,-1.982]],$
[[1.6529,-0.0959,-1.99],[1.6529,-0.0879,-1.982]],$
[[0.7252,0.0879,-1.982],[0.7252,0.0959,-1.99]],$
[[1.6529,0.0879,-1.982],[1.6529,0.0959,-1.99]],$

[[1.204,0.0,-1.982],[1.204,0.0,-1.982]],$ 
[[0.7252,-0.0879,-1.982],[0.7252,-0.0879,-1.982]],$
[[1.3437,-0.0879,-1.982],[1.3437,-0.0879,-1.982]],$
[[0.7252,-0.0959,-2.170],[0.7252,-0.0959,-2.170]],$
[[1.2697,-0.0959,-2.160],[1.2697,-0.0959,-2.160]],$

[[1.017,-0.0879,-1.982],[1.017,0.0879,-1.982]],$
[[1.017,0.0879,-1.982],[1.017,0.0959,-1.99]],$
[[1.017,0.0959,-1.99],[1.017,0.0959,-2.170]],$
[[1.017,0.0959,-2.170],[1.017,-0.0959,-2.170]],$
[[1.017,-0.0959,-2.170],[1.017,-0.0959,-1.99]],$
[[1.017,-0.0959,-1.99],[1.017,-0.0879,-1.982]],$

[[1.333,-0.0879,-1.982],[1.333,0.0879,-1.982]],$
[[1.333,0.0879,-1.982],[1.333,0.0959,-1.99]],$
[[1.333,0.0959,-1.99],[1.333,0.0959,-2.170]],$
[[1.333,0.0959,-2.170],[1.333,-0.0959,-2.170]],$
[[1.333,-0.0959,-2.170],[1.333,-0.0959,-1.99]],$
[[1.333,-0.0959,-1.99],[1.333,-0.0879,-1.982]]$
]

;rotate to lower box locations (60deg-sector 11 and 180deg-sector7)
ia=0
for ang=60.0,180.0,120.0 do begin
	qrot=q*0.0
	for i=0,n_elements(q(0,0,*))-1 do begin
		p=q(*,*,i)
		rot,p,ang,/z
		qrot(*,*,i)=p
	endfor
boxs(*,*,*,ia)=qrot
ia=ia+1
endfor

;define a single upper box
q(2,*,*)=-q(2,*,*)
q(1,*,*)=-q(1,*,*)

;rotate to upper box locations (180deg and 300deg)
ia=0
for ang=180.0,300.0,120.0 do begin
	qrot=q*0.0
	for i=0,n_elements(q(0,0,*))-1 do begin
		p=q(*,*,i)
		rot,p,ang,/z
		qrot(*,*,i)=p
	endfor
boxs(*,*,*,ia+2)=qrot
ia=ia+1
endfor

end

function pf,rcentre,zcentre,dr,dz
;FL:Adds in PF coil pair of given dimensions and position
pf=fltarr(3,2,36,8);PF coils made of 4 rings

drby2=dr/2
dzby2=dz/2

;Using for loop to move around the four corners of upper coil
for corner=0,3 do begin
  r=[rcentre+(-1)^(corner/2)*drby2,0.0,zcentre+(-1)^((corner+1)/2)*dzby2]
  p=r
  ia=0
  for i=0.0,350.0,10.0 do begin
    rot,p,10.0,/z
    pf[*,*,ia,corner]=[[r],[p]]
    r=p
    ia=ia+1
  endfor
endfor
;Now copy for lower coil
for corner=4,7 do begin
  pf[*,*,*,corner]=pf[*,*,*,corner-4];copy from upper
  pf[2,*,*,corner]=-1*pf[2,*,*,corner-4];flip z elements
endfor

return,pf
end

pro get_pf,p3,p4,p5,p6 ;Inputs the coords of the PF coils into the pf function
p3=pf(1.1,1.1,0.12,0.134)
p4=pf(1.5,1.1,0.19,0.19)
p5=pf(1.65,0.5,0.19,0.19)
p6=pf(1.442,0.9,0.058,0.155)
end

pro get_p2mid,p2;New shape P2 armour for MID

;define P2 vector for 2 P2 plates composed of 6 rings of 36 vectors each
p2=fltarr(3,2,36,12)
;define coords in r,z
  rz=[[.290,-1.696],[.290,-1.673],$;Inner edge
      [.416,-1.547],[.577,-1.547],$;Top surface
      [.783,-1.716],[.783,-1.728]];Outer edge
for n=0,5 do begin
  r=[rz(0,n),0.0,rz(1,n)]
  p=r
  ia=0
  for i=0.0,350.0,10.0 do begin
    rot,p,10.0,/z
    p2(*,*,ia,n)=[[r],[p]]
    r=p
    ia=ia+1
  endfor
endfor
p2(0:1,*,*,6:11)=p2(0:1,*,*,0:5)
p2(2,*,*,6:11)=-p2(2,*,*,0:5);Mirror top/bottom by reversing z component
end

pro get_p2,p2

;DEFINE P2

;define P2 vector for 2 P2 plates composed of 6 rings of 36 vectors each
p2=fltarr(3,2,36,12)

;define vector arcs in lower P2 and rotate 2*pi

r=[0.746,0.0,-1.478]
p=r
ia=0
for i=0.0,350.0,10.0 do begin
	rot,p,10.0,/z
	p2(*,*,ia,0)=[[r],[p]]
	r=p
	ia=ia+1
endfor

r=[0.746,0.0,-1.483]
p=r
ia=0
for i=0.0,350.0,10.0 do begin
	rot,p,10.0,/z
	p2(*,*,ia,1)=[[r],[p]]
	r=p
	ia=ia+1
endfor

r=[0.7115,0.0,-1.513]
p=r
ia=0
for i=0.0,350.0,10.0 do begin
	rot,p,10.0,/z
	p2(*,*,ia,2)=[[r],[p]]
	r=p
	ia=ia+1
endfor

r=[0.755,0.0,-1.453]
p=r
ia=0
for i=0.0,350.0,10.0 do begin
	rot,p,10.0,/z
	p2(*,*,ia,3)=[[r],[p]]
	r=p
	ia=ia+1
endfor

r=[0.250,0.0,-1.453]
p=r
ia=0
for i=0.0,350.0,10.0 do begin
	rot,p,10.0,/z
	p2(*,*,ia,4)=[[r],[p]]
	r=p
	ia=ia+1
endfor

r=[0.250,0.0,-1.478]
p=r
ia=0
for i=0.0,350.0,10.0 do begin
	rot,p,10.0,/z
	p2(*,*,ia,5)=[[r],[p]]
	r=p
	ia=ia+1
endfor

;define vector arcs in lower P2 and rotate 2*pi

r=[0.746,0.0,1.478]
p=r
ia=0
for i=0.0,350.0,10.0 do begin
	rot,p,10.0,/z
	p2(*,*,ia,6)=[[r],[p]]
	r=p
	ia=ia+1
endfor

r=[0.746,0.0,1.483]
p=r
ia=0
for i=0.0,350.0,10.0 do begin
	rot,p,10.0,/z
	p2(*,*,ia,7)=[[r],[p]]
	r=p
	ia=ia+1
endfor

r=[0.7115,0.0,1.513]
p=r
ia=0
for i=0.0,350.0,10.0 do begin
	rot,p,10.0,/z
	p2(*,*,ia,8)=[[r],[p]]
	r=p
	ia=ia+1
endfor

r=[0.755,0.0,1.453]
p=r
ia=0
for i=0.0,350.0,10.0 do begin
	rot,p,10.0,/z
	p2(*,*,ia,9)=[[r],[p]]
	r=p
	ia=ia+1
endfor

r=[0.250,0.0,1.453]
p=r
ia=0
for i=0.0,350.0,10.0 do begin
	rot,p,10.0,/z
	p2(*,*,ia,10)=[[r],[p]]
	r=p
	ia=ia+1
endfor

r=[0.250,0.0,1.478]
p=r
ia=0
for i=0.0,350.0,10.0 do begin
	rot,p,10.0,/z
	p2(*,*,ia,11)=[[r],[p]]
	r=p
	ia=ia+1
endfor

end

pro get_cc,cc,mid=mid

;specify differences between original CC and the one with the MID
if keyword_set(mid) then begin
  rarm=0.280;new armour radius
  zeds=[-1.481,-1.229,-1.083];new vertex heights
endif else begin
  rarm=0.211;old armour radius
  zeds=[-1.3725,-1.1995,-1.0675];old vertex heights
endelse

;DEFINE CC

;define CC vector for 12 tile edges of 36 vectors each
cc=fltarr(3,2,36,12)

;define vector arcs and rotate 2*pi

r=[rarm,0.0,zeds(0)];-1.3725 for old CC, -1.481 for new
p=r
ia=0
for i=0.0,350.0,10.0 do begin
	rot,p,10.0,/z
	cc(*,*,ia,0)=[[r],[p]]
	r=p
	ia=ia+1
endfor

r=[rarm,0.0,zeds(1)];-1.1995 for old CC, -1.229 for new
p=r
ia=0
for i=0.0,350.0,10.0 do begin
	rot,p,10.0,/z
	cc(*,*,ia,1)=[[r],[p]]
	r=p
	ia=ia+1
endfor

r=[0.196,0.0,zeds(2)];-1.0675 for old, -1.083 for new
p=r
ia=0
for i=0.0,350.0,10.0 do begin
	rot,p,10.0,/z
	cc(*,*,ia,2)=[[r],[p]]
	r=p
	ia=ia+1
endfor
r=[0.196,0.0,-0.7625]
p=r
ia=0
for i=0.0,350.0,10.0 do begin
	rot,p,10.0,/z
	cc(*,*,ia,3)=[[r],[p]]
	r=p
	ia=ia+1
endfor

r=[0.196,0.0,-0.4575]
p=r
ia=0
for i=0.0,350.0,10.0 do begin
	rot,p,10.0,/z
	cc(*,*,ia,4)=[[r],[p]]
	r=p
	ia=ia+1
endfor

r=[0.196,0.0,-0.1525]
p=r
ia=0
for i=0.0,350.0,10.0 do begin
	rot,p,10.0,/z
	cc(*,*,ia,5)=[[r],[p]]
	r=p
	ia=ia+1
endfor
cc(0:1,*,*,6:11)=cc(0:1,*,*,0:5)
cc(2,*,*,6:11)=-cc(2,*,*,0:5);Mirror top/bottom by reversing z component
end
