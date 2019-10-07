pro finish_o,pr=pr,fi=fi,psfn=psfn,pdev=pdev

if NOT keyword_set(psfn) then psfn='idltmp.ps'
if NOT keyword_set(pr) then pr = 0
if NOT keyword_set(fi) then fi = 0
if NOT keyword_set(pdev) then pdev = 'lw'


if fi then begin
	device,/close
;	set_plot,'WIN' 
	set_plot,'X'
	print,'file output to : ',psfn,' is ready'
	if pr then begin
		print,'printing: lpr -P'+pdev+' '+psfn
		spawn,'lpr -P'+pdev+' '+psfn
	end
end else  wset,0


end
