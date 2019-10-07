;$Header: /funsrv1/home/gdt/Development/code_lwir/load_knl.pro,v 1.1 2003/11/20 10:19:45 akirk Exp $
;$Date: 2003/11/20 10:19:45 $
;$Author: akirk $
;$Locker:  $
;$Log: load_knl.pro,v $
;Revision 1.1  2003/11/20 10:19:45  akirk
;Initial revision
;
;
pro load_knl

	theo_save='THEODOR/theo_knl.sav'
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
