import pytest
import unittest
from pathlib import Path

import numpy as np

from fire.interfaces.calcam_calibs import get_calcam_calib_path_fn

pwd = Path(__file__).parent

class TestCalcam(unittest.TestCase):

    def test_get_calcam_calib_path_fn(self):
        inputs = {'pulse': 30378, 'camera': 'rit', 'machine': 'MAST'}
        out = get_calcam_calib_path_fn(**inputs)
        expected = 'MAST-rit-p23586-n217-enhanced_1-rough_test.ccc'
        self.assertEqual(out, expected)

        inputs = {'pulse': 30378, 'camera': 'rbb', 'machine': 'MAST'}
        with self.assertRaises(FileNotFoundError):
            get_calcam_calib_path_fn(**inputs)

        inputs = {'pulse': 999999, 'camera': 'rit', 'machine': 'MAST'}
        with self.assertRaises(ValueError):
            get_calcam_calib_path_fn(**inputs)


def suite():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    # suite.addTests(loader.loadTestsFromTestCase(TestIoIpx))
    suite.addTest(TestCalcam.test_get_calcam_calib_path_fn)
    return suite

if __name__ == '__main__':

    runner = unittest.TextTestRunner(failfast=True)
    unittest.main()
    # runner.run(suite())