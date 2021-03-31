#!/usr/bin/env python

"""

"""

import unittest
import pickle
from pathlib import Path

import numpy as np

from fire.camera_tools.camera_shake import (calc_camera_shake_displacements, remove_camera_shake)
from fire import fire_paths

pwd = Path(__file__).parent

class TestCamShake(unittest.TestCase):

    def setUp(self):
        path_fn = fire_paths['root'] / '../tests/test_data/frames_with_shake.p'
        with open(path_fn, 'rb') as f:
            self.shaky_frames = pickle.load(f)
        self.expected_displacements = {
                                 None: np.array([[ 0.00000000e+00,  0.00000000e+00],
                                                 [-1.09393997e-02,  5.05917151e-02],
                                                 [-1.14394936e-01,  3.32541698e-01],
                                                 [-2.01143945e+01, -3.96674579e+01],
                                                 [-1.59583610e-02,  3.16369348e-02]]),
                                 10: np.array([[ 0.      ,  0.      ],
                                               [-0.010939,  0.050592],
                                               [-0.114395,  0.332542],
                                               [   np.nan,    np.nan],
                                               [-0.015958,  0.031637]])
        }

    def test_calc_camera_shake_displacements(self):
        frames = self.shaky_frames
        frame_reference = frames[0]

        pixel_displacemnts, shake_stats = calc_camera_shake_displacements(frames, frame_reference,
                                                                          erroneous_displacement=None)
        self.assertTrue(isinstance(shake_stats, dict))
        np.testing.assert_array_almost_equal(pixel_displacemnts, self.expected_displacements[None])


        pixel_displacemnts, shake_stats = calc_camera_shake_displacements(frames, frame_reference,
                                                                          erroneous_displacement=10)
        self.assertTrue(isinstance(shake_stats, dict))
        np.testing.assert_array_almost_equal(pixel_displacemnts, self.expected_displacements[10])

    def test_remove_camera_shake(self):
        frames = self.shaky_frames
        pixel_displacemnts = self.expected_displacements[None]
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