;$Header: /funsrv1/home/gdt/Development/New\040LWIR\040code/gc_fclose.pro,v 1.1 2003/11/20 10:21:25 akirk Exp $
;$Date: 2003/11/20 10:21:25 $
;$Author: akirk $
;$Locker:  $
;$Log: gc_fclose.pro,v $
;Revision 1.1  2003/11/20 10:21:25  akirk
;Initial revision
;
;
function gc_fclose, file, help=help
@ida_file

ok=ida_close(file)
return, ok

end
