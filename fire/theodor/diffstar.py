from numpy import sqrt

def diffstar(h_pot, cc, plusac, tratio, ad, bd):
	hpc = h_pot-cc
	ww = sqrt(hpc*hpc+2.0*h_pot)
	if not plusac: 
		ww *= -1.0
	d_star = 1.0+tratio*(hpc+ww)
	return ad+bd/(d_star*d_star)
