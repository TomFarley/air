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
