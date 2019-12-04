from numpy import sqrt

def weight(h_pot, hts, cc, plusac):
	hpc = h_pot-cc
	ww = sqrt((hpc*hpc)+2.0*h_pot)
	if not plusac: 
		ww *= -1.0
	return ww/(3.0*ww+hts*(ww+hpc+1.0))
