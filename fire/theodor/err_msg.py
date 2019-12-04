def err_msg(error, halt=False):
	max_error = 9
	if error < 0: 
		error = max_error
	error_str = [
		['data values for the fit are increasing'],
		['data values for the fit are constant and negative'],
		['curvature of the fit data is concave'],
		['fit is limited to a maximum temperature'],
		['could not read valid data from file'],
		['data are not of type temperature'],
		['error 7'],
		['number of meshes too low (increase thickness or time resolution)'],
		['size of time and data array differs'],
		['unknown error code']
		]
	if error<=max_error:
		print(error_str[error-1])
	else:
		print(error_str[max_error])
	if halt:
		print("")
		print("--------------------------------------------------------------")
		print("!err_msg: (THEODOR) the HALT-flag is set. Continuing anyway...")
		print("--------------------------------------------------------------")
		print("")
