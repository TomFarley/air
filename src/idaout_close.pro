
	pro idaout_close,fid,shotnr,errorflg

@ida3
  fprefix = 'air'
  fshot=STRING(FORMAT='(i6.6)', shotnr)

;write out the SVN revision number
	spawn, /SH, "svn info . | grep -i revision | cut -d ':' -f2-2", res
	res=strtrim(res,2)
	item=ida_create(fid, fprefix+'_svn_revision', fshot)
	ok=ida_set_structure(item, ida_dct, 1, ida_d4+ida_real, xtsams=nelem0)
	ok=ida_set_dinfo(item, 0.0, 1.0, 0, 1.0, 0.0, 'SVN revision','SVN revision')
	ok=ida_set_data(item, res, ida_d4+ida_real+ida_valu)
	err=GC_FCLOSE(fid)

   s = [0]
   s(0)=errorflg
   item = ida_create(fid, fprefix+'_errorflag', fshot)
   ok = ida_set_structure(item, ida_dct, 1, ida_d4+ida_real, xtsams=nelem)
   ok = ida_set_dinfo(item, 0.0, 1.0, 0, 1.0, 0.0, 'err', 'err')
   ok = ida_set_data(item, s, ida_d4+ida_real+ida_valu)
   err = GC_FCLOSE(fid) 

end

