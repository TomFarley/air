;gegebenenfalls Ausgabe in eine Datei, s/w oder farbig

; A. Herrmann, July/2007

pro init_out,psfn=psfn,co=co,fi=fi,size=size,title=title,gamma=gamma,lasc=lasc,silent=silent

COMMON colors, r_orig,g_orig,b_orig,r_curr,g_curr,b_curr



case chk_sys() of
     'PC': path='D:'
     'UNIX': path='~/'
     ELSE : path=''
 endcase

	if NOT keyword_set(psfn) then psfn=path+'idltmp.ps'
	if NOT keyword_set(fi) then fi = 0
	if NOT keyword_set(co) then if fi eq 0 then co=27 else co=26
	if NOT keyword_set(gamma) then gamma = 1
	if NOT keyword_set(size) then size = 0
	if NOT keyword_set(lasc) then lasc = 0 else lasc = 1
	if NOT keyword_set(title) then title='IDL 0'
	if NOT keyword_set(silent) then silent = 0

;gegebenenfalls Ausgabe in eine Datei, s/w oder farbig

	if fi then begin
        	set_plot,'PS'
        	!P.font=0
        	if size eq 1 then if lasc eq 0 then $
		     begin
		    	    print,'portrait'
		    	    device,filename=psfn,color=co,bits_per_pixel=8,$
		    	    	yoffset=2,ysize=25.7
		     end else begin
		     	    print,'landscape'
		     	    device,filename=psfn,color=co,bits_per_pixel=8,/landscape
		     end else device,filename=psfn,color=co,bits_per_pixel=8,ysize=12.0
     	  END else begin
        	!P.font=-1
        	if chk_sys() eq 'PC' then set_plot,'WIN' else set_plot,'X'
        	if (((!D.Name EQ 'X') OR (!D.NAME EQ 'MAC')) AND $
    			(!D.N_Colors GE 256L)) then DEVICE, PSEUDO_COLOR=8
 ; this line is essential for windows to handle colors
		DEVICE, Decomposed=0
		DEVICE, GET_SCREEN_SIZE = scrsize
     	endelse

	if (co gt 0) then begin
	        loadct,co
  	        tvlct,r_curr,g_curr,b_curr
   	end else begin
        	loadct,0
       	end

	if NOT fi  then $
		if NOT(silent) then $
	    	if size then $
	    		if not lasc then window, 0,xsize=650,ysize=850,title=title $
		    		else window, 0,xsize=850,ysize=650,title=title $
     else window, title=title

	if gamma ne 1 then gamma_ct,gamma
end
