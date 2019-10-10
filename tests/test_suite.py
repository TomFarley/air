import unittest

# from .test_uda_io import TestUdaIO
from .test_ipx_io import TestIpxIO

def main():
    unittest.TextTestRunner(verbosity=3).run(unittest.TestSuite())

if __name__ == '__main__':
    unittest.main()