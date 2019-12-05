import unittest
from pathlib import Path

import numpy as np

from fire.interfaces.movie_plugins.ipx import get_freia_ipx_path, read_movie_meta, read_movie_data

pwd = Path(__file__).parent
ipx_path = (pwd / '../test_data/mast/').resolve()
print(f'pwd: {pwd}')
print(f'ipx test files path: {ipx_path}')

# @pytest.fixture  # Run function once and save output to supply to multiple tests
# def expected_ouput():
# 	aa=25
# 	bb =35
# 	cc=45
# 	return [aa,bb,cc]

class TestIoIpxFast(unittest.TestCase):

    def test_get_freia_ipx_path(self):
        pulse = 30378
        camera = 'rir'
        path = get_freia_ipx_path(pulse, camera)
        path_expected = '/net/fuslsa/data/MAST_IMAGES/030/30378/rir030378.ipx'
        self.assertEqual(path, path_expected)

    def test_get_ipx_meta_data_rir030378(self):
        ipx_fn = 'rir030378.ipx'
        ipx_path_fn = ipx_path / ipx_fn
        ipx_meta_data = read_movie_meta(ipx_path_fn)

        self.assertTrue(isinstance(ipx_meta_data, dict))
        self.assertEqual(len(ipx_meta_data), 10)

        self.assertTrue(np.all(ipx_meta_data['frame_range'] == np.array([0, 3749])))
        self.assertTrue(np.all(ipx_meta_data['t_range'] == np.array([-0.049970999999999995, 0.699828])))
        self.assertEqual(ipx_meta_data['frame_shape'], (8, 320))
        self.assertAlmostEqual(ipx_meta_data['fps'], 5000.006668453812)

        ipx_header_expected = {
            'size': 286, 'codec': 'JP2', 'shot': 30378, 'trigger': -0.10000000149011612, 'lens': '50mm',
            'filter': 'LP4500nm', 'view': 'Lower divertor view#6',
            'camera': 'SBF125 InSb FPA 320x256 format with SBF1134 4Chan Rev6 (1 outpu', 'width': 320, 'height': 8,
            'depth': 14, 'orient': 0, 'taps': 4, 'left': 1, 'right': 320, 'top': 185, 'bottom': 192, 'exposure': 28,
            'strobe': 0, 'board_temp': 50.5, 'ccd_temp': 73.47895050048828, 'n_frames': 3750, 'is_color': 0,
            'ipx_version': 'IPX 01', 'hbin': 0, 'vbin': 0, 'datetime': '23/09/2013 15:22:20', 'preexp': 28,
            'gain': [2.0,2.0], 'offset': [170, 170]
        }
        self.assertEqual(ipx_meta_data['ipx_header'], ipx_header_expected)

    def test_get_ipx_meta_data_rit030378(self):
        ipx_fn = 'rit030378.ipx'
        ipx_path_fn = ipx_path / ipx_fn
        ipx_meta_data = read_movie_meta(ipx_path_fn)

        self.assertTrue(isinstance(ipx_meta_data, dict))
        self.assertEqual(len(ipx_meta_data), 10)

        self.assertTrue(np.all(ipx_meta_data['frame_range'] == np.array([0, 623])))
        self.assertTrue(np.all(ipx_meta_data['t_range'] == np.array([-0.048749,  0.69885 ])))
        self.assertEqual(ipx_meta_data['frame_shape'], (32, 256))
        self.assertAlmostEqual(ipx_meta_data['fps'], 833.3344480129053)

        ipx_header_expected = {
            'width': 256, 'height': 32, 'depth': 14, 'codec': 'jp2', 'datetime': '2013-09-23T15:37:29', 'shot': 30378,
            'trigger': -0.5, 'view': 'HL01 Upper divertor view#1', 'camera': 'Thermosensorik CMT 256 SM HS',
            'top': 153, 'bottom': 184, 'offset': 0.0, 'exposure': 50.0, 'ccdtemp': 59.0, 'frames': 625, 'size': 239,
            'n_frames': 625, 'ipx_version': 'IPX 02'}

        self.assertEqual(ipx_meta_data['ipx_header'], ipx_header_expected)

    def test_get_ipx_movie_data_rir030378(self):
        # Ipx 1 file
        ipx_fn = 'rir030378.ipx'
        ipx_path_fn = ipx_path / ipx_fn

        # # Read whole movie
        # frame_nos, frame_times, frame_data = read_movie_data_ipx(ipx_path_fn)
        # self.assertTrue(isinstance(frame_data, np.ndarray))
        # self.assertTrue(isinstance(frame_nos, np.ndarray))
        # self.assertTrue(isinstance(frame_times, np.ndarray))
        # self.assertEqual(frame_data.shape, (3750, 8, 320))
        # self.assertEqual(frame_times.shape, (3750,))
        # self.assertEqual(frame_nos.shape, (3750,))
        # np.testing.assert_array_equal(frame_nos[[0, -1]], (0, 3749))
        # np.testing.assert_allclose(frame_times[[0, -1]], (-0.049971, 0.699828))
        # frame_data_expected = np.array([[[591, 590, 585], [590, 590, 590], [590, 595, 591]],
        #                                 [[593, 596, 586], [590, 595, 594], [592, 602, 594]]])
        # np.testing.assert_array_equal(frame_data[[895, 1597], ::3, ::150], frame_data_expected)

        # Read specific frames
        frames = [5, 150, 177, 1595, 3749]
        nframes = len(frames)
        frame_nos, frame_times, frame_data = read_movie_data(ipx_path_fn, frame_nos=frames)
        self.assertTrue(isinstance(frame_data, np.ndarray))
        self.assertTrue(isinstance(frame_nos, np.ndarray))
        self.assertTrue(isinstance(frame_times, np.ndarray))
        self.assertEqual(frame_data.shape, (nframes, 8, 320))
        self.assertEqual(frame_times.shape, (nframes,))
        self.assertEqual(frame_nos.shape, (nframes,))
        np.testing.assert_array_equal(frame_nos, frames)
        np.testing.assert_allclose(frame_times, [-0.048971, -0.019971, -0.014571,  0.269028,  0.699828])
        frame_data_expected = np.array([[[592., 589., 588.], [588., 594., 590.], [590., 593., 589.]],
                                        [[590., 596., 592.], [594., 594., 595.], [590., 599., 592.]]])
        np.testing.assert_array_equal(frame_data[[1, 3], ::3, ::150], frame_data_expected)

        # Read single frame
        frames = 2678
        nframes = 1
        frame_nos, frame_times, frame_data = read_movie_data(ipx_path_fn, frame_nos=frames)
        self.assertTrue(isinstance(frame_data, np.ndarray))
        self.assertTrue(isinstance(frame_nos, np.ndarray))
        self.assertTrue(isinstance(frame_times, np.ndarray))
        self.assertEqual(frame_data.shape, (nframes, 8, 320))
        self.assertEqual(frame_times.shape, (nframes,))
        self.assertEqual(frame_nos.shape, (nframes,))
        np.testing.assert_array_equal(frame_nos, frames)
        np.testing.assert_array_equal(frame_times, [0.485628])
        frame_data_expected = np.array([[594., 660., 586.], [589., 662., 591.], [594., 660., 594.]])
        np.testing.assert_array_equal(frame_data[0, ::3, ::150], frame_data_expected)

        with self.assertRaises(TypeError):
            read_movie_data(None, frame_nos=frames)
        with self.assertRaises(FileNotFoundError):
            read_movie_data('not a path', frame_nos=frames)
        with self.assertRaises(ValueError):
            read_movie_data(ipx_path_fn, frame_nos=np.linspace(15, 30, 20))

        # TODO test transforms

# TODO: Add test for ipx2 frame data


class TestIoIpxSlow(unittest.TestCase):

    def test_get_ipx_movie_data_rir030378(self):
        # Ipx 1 file
        ipx_fn = 'rir030378.ipx'
        ipx_path_fn = ipx_path / ipx_fn

        # Read whole movie
        frame_nos, frame_times, frame_data = read_movie_data(ipx_path_fn)
        self.assertTrue(isinstance(frame_data, np.ndarray))
        self.assertTrue(isinstance(frame_nos, np.ndarray))
        self.assertTrue(isinstance(frame_times, np.ndarray))
        self.assertEqual(frame_data.shape, (3750, 8, 320))
        self.assertEqual(frame_times.shape, (3750,))
        self.assertEqual(frame_nos.shape, (3750,))
        np.testing.assert_array_equal(frame_nos[[0, -1]], (0, 3749))
        np.testing.assert_allclose(frame_times[[0, -1]], (-0.049971, 0.699828))
        frame_data_expected = np.array([[[591, 590, 585], [590, 590, 590], [590, 595, 591]],
                                        [[593, 596, 586], [590, 595, 594], [592, 602, 594]]])
        np.testing.assert_array_equal(frame_data[[895, 1597], ::3, ::150], frame_data_expected)

def suite():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    # suite.addTests(loader.loadTestsFromTestCase(TestIoIpx))
    suite.addTest(TestIoIpxFast.test_get_ipx_meta_data_rir030378)
    return suite

if __name__ == '__main__':

    runner = unittest.TextTestRunner(failfast=True)
    # unittest.main()
    runner.run(suite())


