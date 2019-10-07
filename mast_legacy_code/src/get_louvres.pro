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

