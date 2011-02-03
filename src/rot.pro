pro rot,p,ang,x=x,y=y,z=z

dang=ang/360.0*2*!pi

if(keyword_set(x)) then mat=[[1,0,0],[0,cos(dang),sin(dang)],[0,-sin(dang),cos(dang)]]
if(keyword_set(y)) then mat=[[cos(dang),0,sin(dang)],[0,1,0],[-sin(dang),0,cos(dang)]]
if(keyword_set(z)) then mat=[[cos(dang),sin(dang),0],[-sin(dang),cos(dang),0],[0,0,1]]

p=mat#p

end
