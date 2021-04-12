"""
For inmports to resolve, run test suite from top level air directory (not from tests directory) ie:
~/repos/air $ ipython tests/test_suite_fast.py     OR
~/repos/air $ python setup.py test      
"""

import unittest

from tests.unit import test_io_ipx
from tests.unit import test_interfaces, test_calcam, test_utils, test_camera_shake, test_s_coordinate

try:
    import pyuda
    client = pyuda.Client()
    from mast import mast_client
except ImportError as e:
    print(f'Failed to import pyuda: {e}')
    pyuda = False

# initialize the test suite
loader = unittest.TestLoader()
suite  = unittest.TestSuite()

# add tests to the test suite
if pyuda:
    from tests.unit import test_mast_u
    from tests.unit import test_io_uda
    suite.addTests(loader.loadTestsFromTestCase(test_io_uda.TestIoUdaSlow))
    suite.addTests(loader.loadTestsFromTestCase(test_mast_u.TestMastU))


# add tests to the test suite
suite.addTests(loader.loadTestsFromTestCase(test_io_ipx.TestIoIpxSlow))

suite.addTests(loader.loadTestsFromModule(test_calcam))
suite.addTests(loader.loadTestsFromModule(test_camera_shake))
suite.addTests(loader.loadTestsFromModule(test_interfaces))
suite.addTests(loader.loadTestsFromTestCase(test_io_ipx.TestIoIpxFast))
suite.addTests(loader.loadTestsFromModule(test_s_coordinate))
suite.addTests(loader.loadTestsFromModule(test_utils))

# initialize a runner, pass it your suite and run it
runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)