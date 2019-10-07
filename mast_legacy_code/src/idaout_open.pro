
	pro idaout_open,shot=shot,pn=pn,fid                 

@ida3


   shotnr = shot
   shot_int1=shotnr/100
   shot_int2=shotnr-shot_int1*100
   shot_str=STRING(FORMAT='(i4.4, a1, i2.2)', shot_int1, '.', shot_int2)

   fprefix = 'air'

   fname=fprefix+shot_str
;
   fshot=STRING(FORMAT='(i6.6)', shotnr)
;
   PRINT, 'Opening file:', fname,' ', fshot
   fid=GC_FOPEN(fname,/W)

   status = 1
   err = GC_PUT_STATUS(fid, fshot, fprefix, status)
   passnum =0 
   if (KEYWORD_SET(pn)) then passnum = pn
   err = GC_PUT_PASSNO(fid, fshot, fprefix, passnum) 


end

