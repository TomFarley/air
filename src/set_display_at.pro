PRO set_display_at,w=w,b=b,c=c,scon=scon

loadct,0,/silent

common invessel,ribs,boxs,p2,cc

;replace the TVing with cp_shading

common data,d,width,height,xst,yst,comp,dnuc,pix,s
common image,bset,cset,bcut,tcut

e=d-dnuc ;perform NUC correction

;cp_shade, e

if(keyword_set(scon)) then begin
	if(n_elements(b) ne 0) then bset=b ;use vals from slider if set
	if(n_elements(c) ne 0) then cset=c ;by display_event
	bcut=bset/100.0*(max(e)-min(e))+min(e) ;set vals if slider not defined
	tcut=cset/100.0*(max(e)-min(e))+min(e) ;for upper and lower limit

	if(bcut eq 0.0 and tcut eq 0.0) then begin
		print,'Contrast failure - using defaults values'
		bcut=0.0
		tcut=5.0
	endif
endif

;limit the range of the image based on the upper and lower limits set using
;display_event
q=where(e le bcut)
if(q(0) ne -1) then e(q)=bcut
q=where(e ge tcut)
if(q(0) ne -1) then e(q)=tcut
e=(e-bcut)/(tcut-bcut)

;make the image fill the window
e=fix(255*transpose(reverse(transpose(e))))
re=rebin(e,[n_elements(e(*,0))*2,n_elements(e(0,*))*2])

;when set_display is called with w ne 0, then the plot is for the image
; overlayed with the wireframe
;comp contains the wireframe and is made by screen grab of the wireframe only
;ie comp=tvrd()
;If image not full frame, make camera image the same size as comp, with the
;image plotted at the correct x,y offset

if(xst ne 0 or yst ne 0) then begin
	;Make ccd image same size as the wire frame image
	;print, 'Sub array of CCD image'
	re_resized=make_array(640,512) ;define array
	re_resized[*,*]=255 ;set to zero
	;define upper and lower corners of the image
	;xst_up=(xst+n_elements(re[*,0])-1)<511
	;yst_up=(yst+n_elements(re[0,*])-1)<511
	
	;re_resized[xst-2:((xst-2)+width), (yst-height):yst]=re
	;re_resized[left_edge:right_edge, bottom_edge:top_edge]
	re_resized[xst:xst+(width*2)-1, yst:yst+(2*height)-1]=re
	re=re_resized
endif
	

if(n_elements(w) ne 0) then begin
	;ns3 = size(comp)
	;xst_up =xst+n_elements(re(*,0))-1 < ns3(1)-1 ;sets offset for the image
						;in the plot window?

	if(w eq 1) then begin
		;wireframe (in variable comp) is a black on white image. 
		;turn it round so that where it is combined with the IR image
		;wireframe is white
		q=where(comp eq 255)
		r=where(comp eq 0)
		if q[0] ne -1 then comp[q]=0
		if r[0] ne -1 then comp[r]=255
		
		re=comp+re<255

		;re=byte((comp[xst:xst_up,yst:yst+n_elements(re(0,*))-1]+re)<255) ;combines the image and 
										;the wireframe (badly)
	endif
endif

;converting image to 3d array. Related to decomposed=0?
ns = size(re)
re3 = intarr(3,ns[1],ns[2])
for k=0,2 do re3(k,*,*) = re
tv,re3, /true;,xst,yst,/true

;move plot so that centre is 0,0 to match wireframe plotting
;x_plot_ax=findgen(512)-256
;y_plot_ax=findgen(512)-256
;cp_shade, re, x_plot_ax, y_plot_ax

;stop
end

