
;$Header: /funsrv1/home/gdt/Development/code_lwir/gc_put_status.pro,v 1.2 2004/08/13 09:21:21 akirk Exp $
;$Date: 2004/08/13 09:21:21 $
;$Author: akirk $
;$Locker:  $
;$Log: gc_put_status.pro,v $
;Revision 1.2  2004/08/13 09:21:21  akirk
;modified to make it signed
;
;Revision 1.1  2003/11/20 10:23:53  akirk
;Initial revision
;
;
function gc_put_status, file, shotno, prefix, status
@ida_header

nelem=1
s=intarr(nelem)
s(0)=status
item = ida_create(file, prefix+'_STATUS', shotno)
ok = ida_set_structure(item, ida_dct, 1, ida_d4+ida_sgnd+ida_intg, xtsams=nelem)
ok = ida_set_dinfo(item, 0.0, 1.0, 0, 1.0, 0.0, 'Status', 'Status')
ok = ida_set_data(item, s, ida_d4+ida_intg+ida_sgnd+ida_valu)
ok = ida_free(item)

return, ok

end
