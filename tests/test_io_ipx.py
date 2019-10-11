import pytest
import unittest
from pathlib import Path

import numpy as np

from fire.io.ipx import get_ipx_meta_data, get_ipx_frames, return_true

ipx_path = Path('test_data/mast/')

# @pytest.fixture  # Run function once and save output to supply to multiple tests
# def expected_ouput():
# 	aa=25
# 	bb =35
# 	cc=45
# 	return [aa,bb,cc]

class TestIoIpx(unittest.TestCase):


    def test_get_ipx_meta_data_rir030378(self):
        ipx_fn = 'rir030378.ipx'
        ipx_path_fn = ipx_path / ipx_fn
        ipx_meta_data = get_ipx_meta_data(ipx_path_fn)

        self.assertTrue(isinstance(ipx_meta_data, dict))
        self.assertEqual(len(ipx_meta_data), 6)

        self.assertTrue(np.all(ipx_meta_data['frame_range'] == np.array([0, 3749])))
        self.assertTrue(np.all(ipx_meta_data['t_range'] == np.array([-0.049970999999999995, 0.699828])))
        self.assertEqual(ipx_meta_data['frame_shape'], (8, 320))
        self.assertAlmostEqual(ipx_meta_data['fps'], 5001.34035921627)

        ipx_header_expected = {'ID': 'IPX 01', 'size': 286, 'codec': 'JP2', 'date_time': '23/09/2013 15:22:20',
             'shot': 30378, 'trigger': -0.10000000149011612, 'lens': '50mm', 'filter': 'LP4500nm',
             'view': 'Lower divertor view#6', 'numFrames': 3750,
             'camera': 'SBF125 InSb FPA 320x256 format with SBF1134 4Chan Rev6 (1 outpu',
             'width': 320, 'height': 8, 'depth': 14, 'orient': 0, 'taps': 4, 'color': 0, 'hBin': 0, 'left': 1,
             'right': 320, 'vBin': 0, 'top': 185, 'bottom': 192, 'offset_0': 170, 'offset_1': 170, 'gain_0': 2.0,
             'gain_1': 2.0, 'preExp': 28, 'exposure': 28, 'strobe': 0, 'board_temp': 50.5,
             'ccd_temp': 73.47895050048828, 'ipx_version': 1}
        self.assertEqual(ipx_meta_data['ipx_header'], ipx_header_expected)

    def test_get_ipx_meta_data_rit030378(self):
        ipx_fn = 'rit030378.ipx'
        ipx_path_fn = ipx_path / ipx_fn
        ipx_meta_data = get_ipx_meta_data(ipx_path_fn)

        self.assertTrue(isinstance(ipx_meta_data, dict))
        self.assertEqual(len(ipx_meta_data), 6)

        self.assertTrue(np.all(ipx_meta_data['frame_range'] == np.array([0, 623])))
        self.assertTrue(np.all(ipx_meta_data['t_range'] == np.array([-0.048749,  0.69885 ])))
        self.assertEqual(ipx_meta_data['frame_shape'], (32, 256))
        self.assertAlmostEqual(ipx_meta_data['fps'], 834.6720634992823)

        ipx_header_expected = {'ID': 'IPX 02', 'width': 256, 'height': 32, 'depth': 14, 'codec': 'jp2',
                               'datetime': '2013-09-23T15:37:29', 'shot': 30378, 'trigger': -0.5,
                               'view': 'HL01 Upper divertor view#1', 'camera': 'Thermosensorik CMT 256 SM HS',
                               'top': 153, 'bottom': 184, 'offset': 0.0, 'exposure': 50.0, 'ccdtemp': 59.0,
                               'frames': 625, 'size': 239, 'numFrames': 625, 'ipx_version': 2}

        self.assertEqual(ipx_meta_data['ipx_header'], ipx_header_expected)

    # def test_return_true(self):
    #     self.assertTrue(return_true())


def suite():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    # suite.addTests(loader.loadTestsFromTestCase(TestIoIpx))
    suite.addTest(TestIoIpx.test_get_ipx_meta_data_rir030378)
    return suite

if __name__ == '__main__':

    runner = unittest.TextTestRunner(failfast=True)
    # unittest.main()
    runner.run(suite())


