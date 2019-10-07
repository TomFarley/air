
;$Header: /funsrv1/home/gdt/Development/code_lwir/gc_putit.pro,v 1.1 2003/11/20 10:22:42 akirk Exp $
;$Date: 2003/11/20 10:22:42 $
;$Author: akirk $
;$Locker:  $
;$Log: gc_putit.pro,v $
;Revision 1.1  2003/11/20 10:22:42  akirk
;Initial revision
;
;
function gc_putit, file, shotno, name, units, t, data, errors=errors

@ida_header

nelem=n_elements(data)

item = ida_create(file, name, shotno)
if keyword_set(errors) then $
  ok = ida_set_structure(item, ida_dct, 1, ida_d4+ida_real+ida_e4+ida_errb, xtsams=nelem) $
else $
  ok = ida_set_structure(item, ida_dct, 1, ida_d4+ida_real, xtsams=nelem)
ok = ida_set_dinfo(item, 0.0, 1.0, 0, 1.0, 0.0, units, units)
ok = ida_set_data(item, data, ida_d4+ida_real+ida_valu, /by_t)
if keyword_set(errors) then $
  ok = ida_set_errors(item, errors, ida_e4+ida_real+ida_valu, /by_t)
ok = ida_set_tinfo(item, t(0), t(1)-t(0), 's', 'Time')
ok = ida_free(item)

return, ok

end
