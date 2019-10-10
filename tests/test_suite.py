import unittest

import test_ipx_io
import test_uda_io



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
suite.addTests(loader.loadTestsFromModule(test_ipx_io))
suite.addTests(loader.loadTestsFromModule(test_uda_io))

# initialize a runner, pass it your suite and run it
runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)