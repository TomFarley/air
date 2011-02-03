pro set_win,wn,xs=xs,ys=ys
		if !d.name eq 'X' or !d.name eq 'WIN' then begin
		if not keyword_set(xs) then xs =350
		if not keyword_set(ys) then ys =300
;print,wn
;print,xs,ys
		window,wn,xs=xs,ys=ys
	end
end