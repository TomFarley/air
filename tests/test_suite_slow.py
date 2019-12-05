import unittest

from unit import test_io_uda, test_io_ipx
from unit import test_interfaces, test_calcam, test_utils

# initialize the test suite
loader = unittest.TestLoader()
suite  = unittest.TestSuite()

# add tests to the test suite
suite.addTests(loader.loadTestsFromModule(test_io_ipx))
suite.addTests(loader.loadTestsFromModule(test_io_uda))
suite.addTests(loader.loadTestsFromModule(test_interfaces))
suite.addTests(loader.loadTestsFromModule(test_calcam))
suite.addTests(loader.loadTestsFromModule(test_utils))

# initialize a runner, pass it your suite and run it
runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)