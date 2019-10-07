pro load_knl

	theo_save='theo_knl.sav'
	openr,1,theo_save,error=err
	close,1
	if err ne 0 then begin
		print,' this program loads the theodor runtime Kernel from: ',theo_save
		print,'adjust the file path if necessary !!!'
		print,'problems? contact Albrecht.Herrmann@ipp.mpg.de '
    end

	restore,theo_save

	print,'theo_knl  was loaded'

end
