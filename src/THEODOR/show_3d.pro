;***********************************************************************
;
;       Max-Plank-Institut fuer Plasmaphysik (IPP)
;       Boltzmannstr. 2
;       85748 Garching bei Muenchen
;
;       Dr. Albrecht H e r r m a n n
;
;       Tel.: (+49) 89 3299 1388
;
;       Version 1.1 01/12/2000
;
;***********************************************************************
;
; to show data as a top view 3d color plot
;
;****************************************************
pro show_3d,data,time,location,size=size,bot_m=bot_m,xt=xt,yt=yt,zt=zt,pt=pt,$
                noerase=noerase,cs=cs,ll=ll,ul=ul
;****************************************************
common pos_3d, bar_pos, dat_pos, dat1_pos, pt_pos, maxcolor

NP = N_params(0)
if np lt 3 then location=indgen(n_elements(data(0,*)))
if np lt 2 then time=indgen(n_elements(data(*,0)))

; prepare output, picture position, range, colorbar

if NOT keyword_set(bot_m) then bot_m = 0.; bottom margin
if NOT keyword_set(size) then size = 1.;   size of the plot (0...1)
if NOT keyword_set(cs) then cs = 1.    ;  character size
if NOT keyword_set(xt) then xt = 'arbitrary units';
if NOT keyword_set(yt) then yt = 'arbitrary units';
if NOT keyword_set(zt) then zt = 'arbitrary units';
if NOT keyword_set(pt) then pt = 'unknown plot';

if NOT keyword_set(ll) then lll = 0 else lll = 1 ; lower limit of z data range
if NOT keyword_set(ul) then ull = 0 else ull = 1 ; upper limit of z data range
if lll then if ll eq -0.01 then ll=0.

uoff=1
loff=3

if NOT keyword_set(noerase) then erase

maxcolor = !d.n_colors - 1

        bar_x0=0.1*cs & bar_x1=0.17+0.07*(cs-1.)
        dat_x0=0.28+0.1*(cs-1.0) & dat_x1=0.95
        bar_y0=0.1*size*cs + bot_m & bar_y1=0.9*size - 0.02*(cs-1.) + bot_m
        dat_y0=bar_y0           & dat_y1=(bar_y1 - bot_m) + bot_m

        bar_pos=[bar_x0,bar_y0,bar_x1,bar_y1]
        dat_pos=[dat_x0,dat_y0,dat_x1,dat_y1]

        if NOT lll then ll=min(data)
        if NOT ull then ul=max(data)
        ll=float(ll)
        ul=float(ul)

        row = 256
        ;erzeuge Graukeil c(*,*)
        ypos = indgen(row)
        ver = (ul - ll)/(row)
        ypos = ypos*ver + ll
        c = fltarr(10,row)
        xpos=indgen(10)
        for i = 0,9 do c(i:i,*) = ypos

;Darstellung des bearbeiteten Bildes

        shade_surf,c,xpos ,ypos,sh = bytscl(c,top=maxcolor-uoff,MIN=ll,MAX=ul)+loff,$
                AX = 90, AZ = 0,charsize=cs,$
                xs=5,ys=1,zs=4,ytitle=zt,$
                POS=bar_pos,/noerase
		n_time =n_elements(time)
		n_ort = n_elements(location)

        yrange=[location(0),location(n_ort-1)]
         shade_surf,data,time,location,$
         sh = bytscl(data,top=maxcolor-uoff,MIN=ll,MAX=ul)+loff,$
         AX = 90, AZ = 0,ticklen=-0.01,$
         yrange=yrange,$
         charsize=cs,$
         xs=1,ys=1,zs=4,xtitle=xt,ytitle=yt,$
;        xtickformat='(f5.3)',$
         POS=dat_pos,/noerase

         ;Graphiktext
         xyouts,dat_x0,bar_y1+0.01,pt,/normal,$
                 charsize=1.2*cs

end
