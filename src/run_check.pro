PRO run_check, shot, png=png

;program to read in the analysis path and camera data and check that they are aligned.
;based on the check.pro procedure.

;now run leon part, without all of the unneccessary options.
leon_check, shot, [0.0,1.0], nuc=1,t,s,h,err,numsatpix, /disp

if keyword_set(png) then begin
	output=tvrd(true=1)
	filename='MWIR_check_'+strtrim(string(fix(shot)),2)+'.png'
	write_png, filename, output
endif


END
