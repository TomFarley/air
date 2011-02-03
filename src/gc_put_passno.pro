
;$Header: /funsrv1/home/gdt/Development/New\040LWIR\040code/gc_put_passno.pro,v 1.1 2003/11/20 10:23:17 akirk Exp $
;$Date: 2003/11/20 10:23:17 $
;$Author: akirk $
;$Locker:  $
;$Log: gc_put_passno.pro,v $
;Revision 1.1  2003/11/20 10:23:17  akirk
;Initial revision
;
;
function gc_put_passno, file, shotno, prefix, passno
@ida_header

nelem=1
s=intarr(nelem)
s(0)=passno
item = ida_create(file, prefix+'_PASSNUMBER', shotno)
ok = ida_set_structure(item, ida_dct, 1, ida_d4+ida_intg, xtsams=nelem)
ok = ida_set_dinfo(item, 0.0, 1.0, 0, 1.0, 0.0, 'Passno', 'Passno')
ok = ida_set_data(item, s, ida_d4+ida_intg+ida_valu)
ok = ida_free(item)

return, ok

end
