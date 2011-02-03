pro ccd_image_at,o,p,i,j,n,l,ribs,boxs,p2,cc
;draws the wire frame over the image.

;common invessel,ribs,boxs,p2,cc


;FL's simplified ccd_image. Hopefully makes it easier to add new components.
;print, 'CCD_IMAGE_AT'
drawvec_at,ribs,o,p,i,j,n,l
drawvec_at,boxs,o,p,i,j,n,l
drawvec_at,p2,o,p,i,j,n,l
drawvec_at,cc,o,p,i,j,n,l
end

pro drawvec_at,vec,o,p,i,j,n,l 
;FL:added this to tidy up ccd_image
for nr=0,n_elements(vec(0,0,0,*))-1 do begin
  for nv=0,n_elements(vec(0,0,*,0))-1 do begin
    q=vec(*,*,nv,nr)
    draw_at,q,o,i,j,n,l
  endfor
endfor
end

pro draw_at,q,o,i,j,n,l
p=q
for k1=0,n_elements(p(0,*))-1 do begin
k2=(k1+1) mod n_elements(p(0,*))

; interpolate between p(*,k2) and p(*,k1)

dx=(p(0,k2)-p(0,k1))/20.0
dy=(p(1,k2)-p(1,k1))/20.0
dz=(p(2,k2)-p(2,k1))/20.0
more=0
	for b=0,20 do begin
		x=[p(0,k1)+dx*b,p(1,k1)+dy*b,p(2,k1)+dz*b]
		image,x,o,n,i,j,l,r,imult,jmult,err
		if(err eq '') then begin
			if(more eq 0) then begin
 			plots,[imult],[jmult],col=1
				more=1
			endif else plots,[imult],[jmult],col=truecolor('royalblue'),/continue
		endif
	endfor
endfor

end

