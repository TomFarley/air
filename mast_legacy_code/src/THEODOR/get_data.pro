function get_data, shot, exper, data, time, location, fn=fn,$
		 type=type,filter=filter,header=header,path=path

if n_elements(shot) eq 0 then shot=1
shot=string(shot)

str_type=['raw', 'temperature', 'heat flux']
; read the stored parameters
if not keyword_set(fn) then begin
;	fn_ini='get_data.ini'
;	close,1
;	openr,1,fn_ini,error=error
;	if error ne 0 then begin
		if not keyword_set(path) then path=''
		if not keyword_set(filter) then filter='*.*'
;	end else begin
;		print,'reading ini file ',fn_ini
;		path='   '
;		filter='   '
;		readf,1,path
;		readf,1,filter
;		close,1
;	end
	fn=path+shot
end

; read the data from file
	; check wether or not the file exists

	close,1
	openr,1,fn,error=error
	if error ne 0 then begin
		result=pickfile(PATH=path,get_path=path,filter=filter,$
					/must_exist,title='select a binary file')
			if strlen(result) gt 0 then begin
				shot=result
			end else return,5
			fn=shot
	;		openw,1,fn_ini
	;		printf,1,path
	;		printf,1,filter
	;		close,1
	end

    r_file,fn,data,location,time,dat_typ=type,shot=shot,error=error
    ;type=dat_type
	print,'data of type: ',str_type(type)

return,error
end