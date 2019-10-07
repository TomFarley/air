;$Header: /funsrv1/home/gdt/Development/code_lwir/readtxtarray.pro,v 1.8 2005/08/24 10:16:24 fraser Exp $
;$Date: 2005/08/24 10:16:24 $
;$Author: fraser $
;$Locker:  $
;$Log: readtxtarray.pro,v $
;Revision 1.8  2005/08/24 10:16:24  fraser
;Added quiet keyword so non-existence of the file isn't printed
;,
;
;Revision 1.7  2004/09/15 10:41:15  fraser
;Added /pwd option as need to open rir txt file outside of calib dir
;
;Revision 1.6  2004/09/14 13:34:28  fraser
;Added error message (and no return) if file doesn't exist
;Note that as of 1.4, this differs from the version in "Useful Programs"
;
;Revision 1.5  2004/08/20 12:20:23  akirk
;corrected
;
;Revision 1.4  2004/08/20 12:19:20  akirk
;modified to look for dat files in calib directory
;
;Revision 1.3  2004/08/20 11:47:10  akirk
;fls mods
;
;Revision 1.4  2004/07/27 13:28:41  fraser
;No of rows is now an optional parameter (instead of just stopping at EOF)
;If dimensions are not specified, array expands to maximum no of columns
;(previously took the number of columns in the first row)
;
;Revision 1.3  2004/07/27 09:51:44  fraser
;Generalised string reading.  nocolumns parameter now optional.
;Can also specify number of rows to skip (for headings etc)
;Note that as of 1.2, can no longer read CSV files.
;
;Revision 1.2  2004/01/20 14:49:12  fraser
;Modified to work with strings, not just numbers
;
;Revision 1.1  2003/11/20 10:18:47  akirk
;Initial revision
;
;
function readtxtarray, filein, nocolumns,norows,ignore_rows=ignore_rows,pwd=pwd,quiet=quiet
;Version of readtxtarray for strings. Fraser Lott, ImperialCollege/UKAEAFusion
;
;    modify so looks in calib directory
;
;Needed to open file in present working directory (not calib) in some cases
if keyword_set(pwd) then fileopen=filein else fileopen = 'calib/'+filein

if file_test(fileopen) then begin;if file exists then read it
  openr,in,fileopen,/GET_LUN
  row=strarr(1)
  if keyword_set(ignore_rows) then for n=0,ignore_rows-1 do readf,in,row
  readf,in,row ;This will be the first row actually processed
  strrow=(strsplit(row,/extract))
  if keyword_set(nocolumns) then begin
    noinrow=n_elements(strrow)
    array=strarr(nocolumns)     ;This small mod means if nocolumns > strrow,
    if nocolumns ge noinrow then array(0:noinrow-1)=strrow $
      else array=strrow(0:nocolumns-1) ; then spare elements will be zero
  endif else array=strrow
  if keyword_set(norows) then rowsleft=norows-1 else rowsleft=-1
  while eof(in) eq 0 and rowsleft ne 0 do begin
    readf,in,row            ;read another row and graft the new row 
    strrow=(strsplit(row,/extract)) ;onto the bottom of the array
    noinrow=n_elements(strrow)
    if keyword_set(nocolumns) then begin
      strrow2=strarr(nocolumns)
      if nocolumns ge noinrow then strrow2(0:noinrow-1)=strrow $
        else strrow2=strrow(0:nocolumns-1)
    endif else begin;If unspecified, use maximum width as no of columns
      colinarr=n_elements(array(*,0))
      if colinarr ge noinrow then begin
        strrow2=strarr(colinarr)
        strrow2(0:noinrow-1)=strrow
      endif else begin ;If array is not wide enough, add more columns
        rowinarr=n_elements(array(0,*))
        array=[array,strarr(noinrow-colinarr,rowinarr)]
        strrow2=strrow
      endelse
    endelse
    array=[[array],[strrow2]]
    rowsleft=rowsleft-1
  endwhile
  free_lun,in           
  return,array
endif else if not keyword_set(quiet) then print,'File ',fileopen,' does not exist!'
end
