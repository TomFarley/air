import pytest
import unittest
from pathlib import Path

import numpy as np

import pyuda
from fire.interfaces.uda import get_uda_movie_obj, read_movie_meta_uda, read_movie_data_uda

pwd = Path(__file__).parent
ipx_path = (pwd / 'test_data/mast/').resolve()
print(f'pwd: {pwd}')
print(f'ipx test files path: {ipx_path}')

# @pytest.fixture  # Run function once and save output to supply to multiple tests
# def expected_ouput():
# 	aa=25
# 	bb =35
# 	cc=45
# 	return [aa,bb,cc]

class TestIoIpx(unittest.TestCase):

    def test_get_uda_movie_obj(self):
        pulse = 30378
        camera = 'rir'
        n_start, n_end = 100, 110
        vid = get_uda_movie_obj(pulse, camera, n_start=n_start, n_end=n_end)
        self.assertEqual(type(vid), pyuda._video.Video)
        self.assertEqual(vid.camera, 'SBF125 InSb FPA 320x256 format with SBF1134 4Chan Rev6 (1 outpu')
        self.assertEqual(vid.lens, '50mm')

    def test_read_movie_meta_uda_rir030378(self):
        pulse = 30378
        camera = 'rir'

        meta_data = read_movie_meta_uda(pulse, camera)

        self.assertTrue(isinstance(meta_data, dict))
        self.assertEqual(len(meta_data), 6)

        self.assertTrue(np.all(meta_data['frame_range'] == np.array([0, 3749])))
        self.assertTrue(np.all(meta_data['t_range'] == np.array([-0.049970999999999995, 0.699828])))
        self.assertEqual(meta_data['frame_shape'], (8, 320))
        self.assertAlmostEqual(meta_data['fps'], 5000.006668453812)

        ipx_header_expected = {'ID': 'IPX 01', 'size': 286, 'codec': 'JP2', 'date_time': '23/09/2013 15:22:20',
                               'shot': 30378, 'trigger': -0.10000000149011612, 'lens': '50mm', 'filter': 'LP4500nm',
                               'view': 'Lower divertor view#6', 'numFrames': 3750,
                               'camera': 'SBF125 InSb FPA 320x256 format with SBF1134 4Chan Rev6 (1 outpu',
                               'width': 320, 'height': 8, 'depth': 14, 'orient': 0, 'taps': 4, 'color': 0, 'hBin': 0,
                               'left': 1,
                               'right': 320, 'vBin': 0, 'top': 185, 'bottom': 192, 'offset_0': 170, 'offset_1': 170,
                               'gain_0': 2.0,
                               'gain_1': 2.0, 'preExp': 28, 'exposure': 28, 'strobe': 0, 'board_temp': 50.5,
                               'ccd_temp': 73.47895050048828}
        self.assertEqual(meta_data['ipx_header'], ipx_header_expected)

        n_start, n_end = 100, 110
        meta_data = read_movie_meta_uda(pulse, camera, n_start, n_end)

        self.assertTrue(isinstance(meta_data, dict))
        self.assertEqual(len(meta_data), 6)

        self.assertTrue(np.all(meta_data['frame_range'] == np.array([n_start, n_end])))
        self.assertTrue(np.all(meta_data['t_range'] == np.array([-0.049970999999999995, 0.699828])))
        self.assertEqual(meta_data['frame_shape'], (8, 320))
        self.assertAlmostEqual(meta_data['fps'], 5000.006668453812)

        ipx_header_expected = {'ID': 'IPX 01', 'size': 286, 'codec': 'JP2', 'date_time': '23/09/2013 15:22:20',
                               'shot': 30378, 'trigger': -0.10000000149011612, 'lens': '50mm', 'filter': 'LP4500nm',
                               'view': 'Lower divertor view#6', 'numFrames': 3750,
                               'camera': 'SBF125 InSb FPA 320x256 format with SBF1134 4Chan Rev6 (1 outpu',
                               'width': 320, 'height': 8, 'depth': 14, 'orient': 0, 'taps': 4, 'color': 0, 'hBin': 0,
                               'left': 1,
                               'right': 320, 'vBin': 0, 'top': 185, 'bottom': 192, 'offset_0': 170, 'offset_1': 170,
                               'gain_0': 2.0,
                               'gain_1': 2.0, 'preExp': 28, 'exposure': 28, 'strobe': 0, 'board_temp': 50.5,
                               'ccd_temp': 73.47895050048828}
        self.assertEqual(meta_data['ipx_header'], ipx_header_expected)

    def test_read_movie_data_uda_rir030378(self):
        # Ipx 1 file
        pulse = 30378
        camera = 'rir'

        # Read whole movie
        frame_nos, frame_times, frame_data = read_movie_data_uda(pulse, camera)
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

        # Read frame range
        n_start, n_end = 100, 110
        frames = np.arange(n_start, n_end+1)
        nframes = (n_end - n_start) + 1
        frame_nos, frame_times, frame_data = read_movie_data_uda(pulse, camera, n_start=n_start, n_end=n_end)
        self.assertTrue(isinstance(frame_data, np.ndarray))
        self.assertTrue(isinstance(frame_nos, np.ndarray))
        self.assertTrue(isinstance(frame_times, np.ndarray))
        self.assertEqual(frame_data.shape, (nframes, 8, 320))
        self.assertEqual(frame_times.shape, (nframes,))
        self.assertEqual(frame_nos.shape, (nframes,))
        np.testing.assert_array_equal(frame_nos, frames)
        frame_times_expected = [-0.029971, -0.029771, -0.029571, -0.029371, -0.029171, -0.028971,
                                -0.028771, -0.028571, -0.028371, -0.028171, -0.027971]
        np.testing.assert_allclose(frame_times, frame_times_expected)
        frame_data_expected = np.array([[[584., 593., 590.], [591., 596., 579.], [589., 598., 592.]],
                                        [[595., 588., 593.], [592., 592., 590.], [591., 597., 591.]]])
        np.testing.assert_array_equal(frame_data[[1, 3], ::3, ::150], frame_data_expected)

        if False:
            # Read single frame
            n_start, n_end = 2678, 2678
            frames = [2678]
            nframes = 1
            frame_nos, frame_times, frame_data = read_movie_data_uda(pulse, camera, n_start=n_start, n_end=n_end)
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

        # with self.assertRaises(TypeError):
        #     read_movie_data_ipx(None, frame_nos=frames)
        # with self.assertRaises(FileNotFoundError):
        #     read_movie_data_ipx('not a path', frame_nos=frames)
        # with self.assertRaises(ValueError):
        #     read_movie_data_ipx(ipx_path_fn, frame_nos=np.linspace(15, 30, 20))

    def test_get_ipx_meta_data_rit030378(self):
        # Ipx 2 file
        pulse = 30378
        camera = 'rir'
        ipx_meta_data = read_movie_meta_uda(pulse, camera)

        self.assertTrue(isinstance(ipx_meta_data, dict))
        self.assertEqual(len(ipx_meta_data), 6)

        self.assertTrue(np.all(ipx_meta_data['frame_range'] == np.array([0, 623])))
        self.assertTrue(np.all(ipx_meta_data['t_range'] == np.array([-0.048749,  0.69885 ])))
        self.assertEqual(ipx_meta_data['frame_shape'], (32, 256))
        self.assertAlmostEqual(ipx_meta_data['fps'], 833.3344480129053)

        ipx_header_expected = {'ID': 'IPX 02', 'width': 256, 'height': 32, 'depth': 14, 'codec': 'jp2',
                               'datetime': '2013-09-23T15:37:29', 'shot': 30378, 'trigger': -0.5,
                               'view': 'HL01 Upper divertor view#1', 'camera': 'Thermosensorik CMT 256 SM HS',
                               'top': 153, 'bottom': 184, 'offset': 0.0, 'exposure': 50.0, 'ccdtemp': 59.0,
                               'frames': 625, 'size': 239, 'numFrames': 625}

        self.assertEqual(ipx_meta_data['ipx_header'], ipx_header_expected)

    def test_get_uda_movie_data_rir030378(self):
        # Ipx 2 file
        pulse = 30378
        camera = 'rir'

        # Read whole movie
        frame_nos, frame_times, frame_data = read_movie_data_uda(pulse, camera)
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

        # Read frame range
        n_start, n_end = 100, 110
        frames = np.arange(n_start, n_end + 1)
        nframes = (n_end - n_start) + 1
        frame_nos, frame_times, frame_data = read_movie_data_uda(pulse, camera, n_start=n_start, n_end=n_end)
        self.assertTrue(isinstance(frame_data, np.ndarray))
        self.assertTrue(isinstance(frame_nos, np.ndarray))
        self.assertTrue(isinstance(frame_times, np.ndarray))
        self.assertEqual(frame_data.shape, (nframes, 8, 320))
        self.assertEqual(frame_times.shape, (nframes,))
        self.assertEqual(frame_nos.shape, (nframes,))
        np.testing.assert_array_equal(frame_nos, frames)
        frame_times_expected = [-0.029971, -0.029771, -0.029571, -0.029371, -0.029171, -0.028971,
                                -0.028771, -0.028571, -0.028371, -0.028171, -0.027971]
        np.testing.assert_allclose(frame_times, frame_times_expected)
        frame_data_expected = np.array([[[584., 593., 590.], [591., 596., 579.], [589., 598., 592.]],
                                        [[595., 588., 593.], [592., 592., 590.], [591., 597., 591.]]])
        np.testing.assert_array_equal(frame_data[[1, 3], ::3, ::150], frame_data_expected)

        if False:
            # Read single frame
            n_start, n_end = 2678, 2678
            frames = [2678]
            nframes = 1
            frame_nos, frame_times, frame_data = read_movie_meta_uda(pulse, camera, n_start=n_start, n_end=n_end)
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

        # TODO test transforms

# TODO: Add test for ipx2 frame data

def suite():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    # suite.addTests(loader.loadTestsFromTestCase(TestIoIpx))
    # suite.addTest(TestIoIpx.test_get_ipx_meta_data_rir030378)
    return suite

if __name__ == '__main__':

    runner = unittest.TextTestRunner(failfast=True)
    unittest.main()
    # runner.run(suite())


