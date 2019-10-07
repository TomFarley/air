function get_toffset, shot ;FL: Gets value from toffset.dat or rir text file

toffarr=readtxtarray('toffset.dat',ignore_rows=1)
leshot=where(toffarr(0,*) le shot,noel)
if noel gt 0 then begin
  geshot=where(toffarr(1,leshot) ge shot,noel)
  if noel gt 0 then begin
    toffset=toffarr(2,leshot(geshot));Use the overlapping subset
    toffset=float(toffset(n_elements(toffset)-1));if many toffsets, use latest
  endif else begin
    ;print,'No time offset found (possibly shot is too new).  Setting to zero.'
    ;toffset=0
;FL 15/9/2004:Will make life a lot easier to read toffset from rir text file
    toffset=readheadfile(shot,/toffset)
    if toffset eq 0 then print,'Time offset set to zero (probably wrong)'
  endelse
endif else begin
  print,'No time offset found (shot too old).  Setting to zero.'
  toffset=0
endelse
return,toffset
end

