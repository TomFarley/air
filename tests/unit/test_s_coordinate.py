import unittest
from pathlib import Path

import pandas as pd

import calcam

from fire.geometry.s_coordinate import get_nearest_rz_coordinates

pwd = Path(__file__).parent

class TestSCoord(unittest.TestCase):

    def test_get_nearest_rz_coordinates(self):
        raise NotImplementedError

def suite():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    # suite.addTests(loader.loadTestsFromTestCase(TestIoIpx))
    suite.addTest(TestSCoord.test_get_calcam_calib_path_fn)
    return suite

if __name__ == '__main__':

    runner = unittest.TextTestRunner(failfast=True)
    unittest.main()
    # runner.run(suite())