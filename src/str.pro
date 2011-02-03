function str,inval,formstat=formstat
;
;  str(11.111,formstat='(f4.1)')
;

if (not keyword_set(formstat)) then begin
   outstring = strtrim(string(inval),1)
endif else begin
   outstring=STRTRIM(STRING(FORMAT=formstat, inval),1) 

endelse

return, outstring

end
