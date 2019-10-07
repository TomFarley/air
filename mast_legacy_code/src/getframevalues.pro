function getframevalues,framerate,width,height,numframes
; GET the proper frame values

oldwidth=width
oldheight=height
frate=framerate

if(oldwidth eq 320 and oldheight eq 256 and frate gt 315.657) then frate=315.657
; GFC this is a temoporary fudge - change if possible

width=oldwidth
height=round(3.2e7/frate/(width+64)-8)/8*8

if(height gt 256) then height=oldheight
; GFC this is a temoporary fudge - change if possible

numframes=numframes*oldwidth/width*oldheight/height

end
