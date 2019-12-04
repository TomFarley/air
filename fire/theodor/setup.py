
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension("heatpotential", ["heatpotential.pyx"] 
	, extra_compile_args = ['-O3','-lm','-march=native'] 
	, include_dirs = [numpy.get_include()] ) ]

setup(
  name = 'Temperature to heat potential',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)

