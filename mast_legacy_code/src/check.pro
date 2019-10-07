;IDL wrapper for checking LEON alignment. Fraser Lott, 2006
pro check,shot
device,retain=2
device,decompose=0

shot=long(shot)
alignarr=readtxtarray('align.dat',3) ;Read in from two column data file
leshot=where(alignarr(0,*) le shot,noel)
;print,'Using alignment set on shot',max(fix(alignarr(0,leshot)),loc)

iranalysis,shot,[0.0,1.0],alignarr(2,n_elements(leshot)-1),t,s,h,q,ldef,numsatpix,1e9,/disp
stop,'Type .c to run ISP'
iranalysis,shot,[0.0,1.0],alignarr(2,n_elements(leshot)-1),t,s,h,q,ldef,numsatpix,1e9,/disp
end  
