import unittest

import test_io_ipx
import test_io_uda

# from .test_uda_io import TestUdaIO
# from .test_ipx_io import TestIpxIO
# def main():
#     unittest.TextTestRunner(verbosity=3).run(unittest.TestSuite())
#
# if __name__ == '__main__':
#     unittest.main()

# initialize the test suite
loader = unittest.TestLoader()
suite  = unittest.TestSuite()

# add tests to the test suite
suite.addTests(loader.loadTestsFromModule(test_io_ipx))
suite.addTests(loader.loadTestsFromModule(test_io_uda))

# initialize a runner, pass it your suite and run it
runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)