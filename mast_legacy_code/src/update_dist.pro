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

