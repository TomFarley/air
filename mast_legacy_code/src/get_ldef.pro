pro get_ldef,loc,ldef;new version which reads ldef from ldef.dat

ldefarr=readtxtarray('ldef.dat',ignore_rows=1)
here=where(ldefarr(0,*) eq loc,noel)
if noel eq 1 then begin
  if strlen(ldefarr(7,here)) eq 0 then begin ;this is for lines
    ldef=[[ldefarr(1:3,here)],[ldefarr(4:6,here)]]
  endif else begin              ;this is for areas
    ldef=[[[ldefarr(1:3,here)], [ldefarr(4:6,here)]],$
          [[ldefarr(7:9,here)],[ldefarr(10:12,here)]]]
  endelse
  ldef=float(ldef)
endif
end
