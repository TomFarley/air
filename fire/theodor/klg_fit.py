from numpy import sqrt,float32,arccos,cos,pi

def klg_fit(data):
	# Y(t) = a + b / (1 + t/t0)^2
	# with temperatures t and t0
	if data.size != 3:
		raise Exception('Wrong number of datapoints for fit: %d' % data.size)
	data = data.astype(float32)
	y1,y2,y3 = data
	# initialize the variables
	a = 0.0
	b = 0.0
	t0 = 0.0
	t_max = 0.0
	# check limits (errors)
	# increasing ?
	#;if y3 gt y2 and y1 gt y2 then return,1
	#;constant and not negative?
	if y1==y2 and y2==y3:
		t0 = 500.0
		if y1<=0:
			return 2, (a, b, t0, t_max)
		else:
			a = y1
		return 0, (a, b, t0, t_max)
	#convex
	#; if  2*y2 ge y1+y3 then return,3
	#linear
	if  2.0*y2==y1+y3:
		return 3, (a, b, t0, t_max)
	# start the fit
	W = sqrt((y2-y3)/(y1-y3))
	phi = arccos(W)
	chi = cos((pi-phi)/3.0)/W
	t0 = 500.0/(chi-1.0)
	b = chi*chi/(chi*chi-1.0)*(y1-y2)
	a = y1-b
	if a<0:
		t_max = t0*(sqrt(-b/a)-1.0)
		return 4, (a, b, t0, t_max)
	if a==0:
		a = 0.001*y3
	return 0, (a, b, t0, t_max)
