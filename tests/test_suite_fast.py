import unittest

from .unit import test_io_uda
from .unit import test_io_ipx
from .unit import test_interfaces, test_calcam, test_utils

try:
    import pyuda
    client = pyuda.Client()
except ImportError as e:
    print(f'Failed to import pyuda. ')
    pyuda = False

# initialize the test suite
loader = unittest.TestLoader()
suite  = unittest.TestSuite()

# add tests to the test suite
if pyuda:
    suite.addTests(loader.loadTestsFromTestCase(test_io_uda.TestIoUdaFast))
suite.addTests(loader.loadTestsFromTestCase(test_io_ipx.TestIoIpxFast))
# suite.addTests(loader.loadTestsFromModule(test_io_uda))
suite.addTests(loader.loadTestsFromModule(test_interfaces))
suite.addTests(loader.loadTestsFromModule(test_calcam))
suite.addTests(loader.loadTestsFromModule(test_utils))

# initialize a runner, pass it your suite and run it
runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)