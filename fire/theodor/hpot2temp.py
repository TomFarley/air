from numpy import sqrt

def hpot2temp(h_pot, cc, tc0):
	hpc = h_pot-cc
	return tc0*(hpc+sqrt(hpc*hpc+2.0*h_pot))
