function readheadfile,shot,lambda=lambda,toffset=toffset,aug=aug;FL: Reads the rir txt file
;toffset, lambda switches say which variable to output, only one at the mo


   shotstring = strtrim(string(shot),1)
   shotnr = shot
   shot_k=shotnr/1000
   shot_int1=shotnr/100
   shot_int2=shotnr-shot_int1*100
   shot_str=STRING(FORMAT='(i4.4, a1, i2.2)', shot_int1, '.', shot_int2)
   shot_k=STRING(FORMAT='(i3.3)', shot_k)
   fname='rir'+shot_str

    filename='$MAST_DATA/'+shot_k+'/'+shotstring+'/Images/rir0'+shotstring+'.ipx'
    descr = ipx_open(filename)

    a= ipx_frame(descr,0,time=time0)
    b= ipx_frame(descr,1,time=time1)

;
;   note the time of the first frame appears to be the end of it -
;  therefore to return to the time offset that was given in the header file
;   we take the difference between the next two frames to get to a start time
; 
    time_offset = time0; - abs(time0 - time1 )

    print,'time offset =',time_offset

    read_lambda = descr.header.filter

    get_lun,lun
    openr,lun,filename,error=err
    if(err eq 0) then begin
                close,/all
                free_lun,lun


        if keyword_set(lambda) and not keyword_set(toffset) then begin
             return,read_lambda ;Better to return name of filter when using bbtable
        endif else begin
              return,-time_offset;toffset
        endelse

     endif else begin
                free_lun,lun
          if keyword_set(lambda) then return,'none' else return,0
     endelse


end
