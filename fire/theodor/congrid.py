# CONGRID-like functions for Python version of THEODOR
# by Laurent Sartran <lsartran@gmail.com> - Ecole Centrale Paris - July 2011

import numpy
from scipy.interpolate import interp1d, RectBivariateSpline

def congrid1d(array, nx):
	'''Equivalent of IDL's CONGRID with linear interpolation for 1D arrays'''
	try:
		if array.ndim!=1:
			raise Exception("1D arrays only, got %dD" % array.ndim)
		oldx = array.size
		if oldx==nx:
			return array
		xOld = numpy.linspace(0, oldx-1, oldx).astype(array.dtype)
		xNew = numpy.linspace(0, oldx-1, nx).astype(array.dtype)
		return interp1d(xOld, array)(xNew).astype(array.dtype)
	except Exception as Error:
		raise Exception('congrid1d\n\t-> %s' % Error)

def congrid2d(array, nx, ny):
	'''Equivalent of IDL's CONGRID with linear interpolation for 2D arrays'''
	try:
		if array.ndim!=2:
			raise Exception("2D arrays only, got %dD" % array.ndim)
		oldx,oldy=array.shape
		if oldx==nx and oldy==ny:
			return array
		xOld = numpy.arange(oldx)
		yOld = numpy.arange(oldy)
		xNew = numpy.linspace(0, oldx-1, nx).astype(array.dtype)
		yNew = numpy.linspace(0, oldy-1, ny).astype(array.dtype)
		return RectBivariateSpline(xOld, yOld, array, kx=1, ky=1)(xNew, yNew).astype(array.dtype)
	except Exception as Error:
		raise Exception('congrid2d\n\t-> %s' % Error)

