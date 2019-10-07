pro err_msg,error,halt=halt
	max_error=9
	if error le 0 then error=max_error
	error_str=[['data values for the fit are increasing'],$
			['data values for the fit are constant and negative'],$
			['curvature of the fit data is concave'],$
			['fit is limited to a maximum temperature'],$
			['could not read valid data from file'],$
			['data are not of type temperature'],$
			['error 7'],$
			['number of meshes to low (increase thickness or time resolution)'],$
			['size of time and data array differs'],$
			['unknown error code']]
	if error le max_error then print,error_str(error-1) else print,error_str(max_error)
if keyword_set(halt) then stop
end
