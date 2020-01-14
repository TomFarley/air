#!/usr/bin/env python

"""

"""

import unittest
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from fire.camera_shake import (calc_camera_shake_displacements, remove_camera_shake)
from fire import fire_paths

pwd = Path(__file__).parent

class TestInterfaces(unittest.TestCase):

    def setUp(self):
        path_fn = fire_paths['root'] / '../tests/test_data/frames_with_shake.p'
        with open(path_fn, 'rb') as f:
            self.shaky_frames = pickle.load(f)
        self.expected_displacements = np.array([[ 0.00000000e+00, -2.84217094e-14],
                                                [-1.09393959e-02,  5.05923943e-02],
                                                [-1.14395314e-01,  3.32538595e-01],
                                                [-2.01143953e+01, -3.96674614e+01],
                                                [-1.59583918e-02,  3.16370797e-02]])

    def test_calc_camera_shake_displacements(self):
        frames = self.shaky_frames
        frame_reference = frames[0]
        pixel_displacemnts, shake_stats = calc_camera_shake_displacements(frames, frame_reference)
        self.assertTrue(isinstance(shake_stats, dict))
        np.testing.assert_array_almost_equal(pixel_displacemnts, self.expected_displacements)

    def test_remove_camera_shake(self):
        frames = self.shaky_frames
        pixel_displacemnts = self.expected_displacements
        # 4th frame is copy of 3rd frame shifted by (40, 20) pixels using np.roll
        self.assertTrue(np.any(np.not_equal(frames[3, 0:-41, 0:-21], frames[2, 0:-41, 0:-21])))

        frames_corrected = remove_camera_shake(frames, pixel_displacemnts)
        # First frame shouldn't have been shifted at all
        np.testing.assert_array_equal(frames_corrected[0], frames[0])
        # Check artificial shift has been removed
        # 4th frame is copy of 3rd frame shifted by (40, 20) pixels using np.roll
        np.testing.assert_array_equal(frames_corrected[3, 0:-41, 0:-21], frames[2, 0:-41, 0:-21])



def suite():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    # suite.addTests(loader.loadTestsFromTestCase(TestIoIpx))
    # suite.addTest(TestInterfaces.test_update_call_args)
    return suite

if __name__ == '__main__':

    runner = unittest.TextTestRunner(failfast=True)
    unittest.main()
    # runner.run(suite())

if __name__ == '__main__':
    pass