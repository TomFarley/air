import pytest
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import calcam

from fire.interfaces.calcam_calibs import get_calcam_calib_info, get_calcam_calib

pwd = Path(__file__).parent

class TestCalcam(unittest.TestCase):

    def test_get_calcam_calib_info(self):
        inputs = {'pulse': 30378, 'camera': 'rit', 'machine': 'MAST', 'search_paths': pwd/'../test_data'}
        out = get_calcam_calib_info(**inputs)
        expected = pd.Series({'pulse_start': 0,
                             'pulse_end': 49999,
                             'calcam_calibration_file': 'MAST-rit-p23586-n217-enhanced_1-rough_test.ccc',
                             'author': 'Tom Farley',
                             'comments': 'Rough test - Do NOT use'})
        self.assertTrue(isinstance(out, pd.Series))
        self.assertDictEqual(out.to_dict(), expected.to_dict())

        inputs['pulse'] = 4000000
        with self.assertRaises(ValueError):
            out = get_calcam_calib_info(raise_=True, **inputs)
        # inputs = {'pulse': 30378, 'camera': 'rbb', 'machine': 'MAST'}
        # with self.assertRaises(FileNotFoundError):
        #     get_calcam_calib_path_fn(**inputs)
        #
        # inputs = {'pulse': 999999, 'camera': 'rit', 'machine': 'MAST'}
        # with self.assertRaises(ValueError):
        #     get_calcam_calib_path_fn(**inputs)

    def test_get_calcam_calib(self):
        path = pwd / '../test_data/mast'
        fn = 'rir_23586_0079_2.ccc'
        calcam_calib_path_fn = path / fn

        out = get_calcam_calib(calcam_calib_path_fn)
        self.assertTrue(isinstance(out, calcam.Calibration))

        out = get_calcam_calib(fn, calcam_calib_path=path)
        self.assertTrue(isinstance(out, calcam.Calibration))

        out = get_calcam_calib(calcam_calib_path_fn.stem, calcam_calib_path=path)
        self.assertTrue(isinstance(out, calcam.Calibration))


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