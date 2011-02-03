function rtptoxyz,vrtp

vxyz=[vrtp(0)*cos(vrtp(1)/360.0*2*!pi),vrtp(0)*sin(vrtp(1)/360.0*2*!pi),vrtp(2)]

return,vxyz

end

function dot,a,b

c=total(a*b)

return,c

end

function cross,a,b

c=[a(1)*b(2)-a(2)*b(1),a(2)*b(0)-a(0)*b(2),a(0)*b(1)-a(1)*b(0)]

return,c

end

pro rot,p,ang,x=x,y=y,z=z

dang=ang/360.0*2*!pi

if(keyword_set(x)) then mat=[[1,0,0],[0,cos(dang),sin(dang)],[0,-sin(dang),cos(dang)]]
if(keyword_set(y)) then mat=[[cos(dang),0,sin(dang)],[0,1,0],[-sin(dang),0,cos(dang)]]
if(keyword_set(z)) then mat=[[cos(dang),sin(dang),0],[-sin(dang),cos(dang),0],[0,0,1]]

p=mat#p

end

pro trans,p,x

p=p-(x#replicate(1,n_elements(p)))

end

pro norm,i,j,n

n=cross(i,j)
n=n/sqrt(dot(n,n))

end
