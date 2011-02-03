
;$Header: /funsrv1/home/gdt/Development/code_lwir/gc_fopen.pro,v 1.1 2003/11/20 10:21:45 akirk Exp $
;$Date: 2003/11/20 10:21:45 $
;$Author: akirk $
;$Locker:  $
;$Log: gc_fopen.pro,v $
;Revision 1.1  2003/11/20 10:21:45  akirk
;Initial revision
;
;
function gc_fopen, fname, w=w

@ida_file

if keyword_set(w) then begin
  file=ida_open(fname, ida_write)
endif else begin
  file=ida_open(fname, ida_read)
  if (file eq 0) then begin
    print, 'No such file'
  endif else begin
    err=ida_get_finfo(file, name, time, date, machine, type)
    if (err ne 0) then begin
      print, 'Not an ida file'
      file=0
    endif else begin
      print,'File OK, created:', date, time
    endelse
  endelse
endelse  

return, file

end
