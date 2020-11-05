import unittest
from pathlib import Path

import numpy as np

try:
    import pyuda
except ImportError as e:
    print(f'Failed to import pyuda for tests')
    pyuda = None
from fire.plugins.movie_plugins.uda import get_uda_movie_obj_legacy, read_movie_meta, read_movie_data

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

if pyuda is not None:

    class TestIoUdaFast(unittest.TestCase):

        def setUp(self):
            self.maxDiff = None

        def test_get_uda_movie_obj(self):
            pulse = 30378
            camera = 'rir'
            n_start, n_end = 100, 110
            vid = get_uda_movie_obj_legacy(pulse, camera, n_start=n_start, n_end=n_end)
            self.assertEqual(type(vid), pyuda._video.Video)
            self.assertEqual(vid.camera, 'SBF125 InSb FPA 320x256 format with SBF1134 4Chan Rev6 (1 outpu')
            self.assertEqual(vid.lens, '50mm')

        def test_read_movie_meta_uda_rir030378(self):
            pulse = 30378
            camera = 'rir'

            meta_data = read_movie_meta(pulse, camera)

            self.assertTrue(isinstance(meta_data, dict))
            self.assertEqual(len(meta_data), 11)

            self.assertTrue(np.all(meta_data['frame_range'] == np.array([0, 3749])))
            self.assertTrue(np.all(meta_data['t_range'] == np.array([-0.049970999999999995, 0.699828])))
            np.testing.assert_array_equal(meta_data['image_shape'], (8, 320))
            self.assertAlmostEqual(meta_data['fps'], 5000.006668453812)

            ipx_header_expected = {
                'board_temp': 50.5, 'camera': 'SBF125 InSb FPA 320x256 format with SBF1134 4Chan Rev6 (1 outpu',
                'ccd_temp': 73.47895050048828, 'codex': 'JP1',
                'date_time': '2013-10-23T15:22:20Z',
                'depth': 14, 'exposure': 28.0,
                'file_format': 'IPX-1',
                'filter': 'LP4500nm', 'gain': np.array([2., 2.]), 'hbin': 0, 'height': 8, 'is_color': 0, 'left': 1,
                'lens': '50mm', 'n_frames': 3750, 'offset': np.array([170., 170.]),
                'orientation': 0,
                'pre_exp': 28.0,
                'shot': 30378,
                'strobe': 0,
                'taps': 4,
                'trigger': -0.10000000149011612,
                'top': 185, 'vbin': 0, 'view': 'Lower divertor view#6', 'width': 320, 'bottom': 177, 'right': 321
            }
            meta_data['ipx_header'].pop('frame_times')
            # Check keys are identical
            self.assertTrue(sorted(meta_data['ipx_header']) == sorted(ipx_header_expected))
            # Check values match
            self.assertTrue(np.all([np.all(meta_data['ipx_header'][key] == ipx_header_expected[key])
                                    for key in ipx_header_expected]))

            n_start, n_end = 100, 110
            meta_data = read_movie_meta(pulse, camera, n_start, n_end)

            self.assertTrue(isinstance(meta_data, dict))
            self.assertEqual(len(meta_data), 11)

            self.assertTrue(np.all(meta_data['frame_range'] == np.array([n_start, n_end])))
            np.testing.assert_almost_equal(meta_data['t_range'], np.array([-0.029971, -0.027971]))
            np.testing.assert_array_equal(meta_data['image_shape'], (8, 320))
            self.assertAlmostEqual(meta_data['fps'], 5000.006668453812)

            meta_data['ipx_header'].pop('frame_times')
            self.assertTrue(np.all([np.all(meta_data['ipx_header'][key] == ipx_header_expected[key])
                                    for key in ipx_header_expected]))

        def test_read_movie_data_uda_rir030378(self):
            # Ipx 1 file
            pulse = 30378
            camera = 'rir'

            # Read single frame
            n_start, n_end = 2678, 2678
            frames = [2678]
            nframes = 1
            frame_nos, frame_times, frame_data = read_movie_data(pulse, camera, n_start=n_start, n_end=n_end)
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

        def test_read_movie_meta_uda_rit030378(self):
            # Ipx 2 file
            pulse = 30378
            camera = 'rit'
            ipx_meta_data = read_movie_meta(pulse, camera)

            self.assertTrue(isinstance(ipx_meta_data, dict))
            self.assertEqual(len(ipx_meta_data), 11)

            self.assertTrue(np.all(ipx_meta_data['frame_range'] == np.array([0, 624])))
            self.assertTrue(np.all(ipx_meta_data['t_range'] == np.array([-0.049949,  0.69885])))
            np.testing.assert_array_equal(ipx_meta_data['image_shape'], (32, 256))
            self.assertAlmostEqual(ipx_meta_data['fps'], 833.3344480129053, places=5)

            # ipx_header_expected = {'ID': 'IPX 02', 'width': 256, 'height': 32, 'depth': 14, 'codec': 'jp2',
            #                        'datetime': '2013-09-23T15:37:29', 'shot': 30378, 'trigger': -0.5,
            #                        'view': 'HL01 Upper divertor view#1', 'camera': 'Thermosensorik CMT 256 SM HS',
            #                        'top': 153, 'bottom': 184, 'offset': 0.0, 'exposure': 50.0, 'ccdtemp': 59.0,
            #                        'frames': 625, 'size': 239, 'n_frames': 625}
            # TODO: Understand why datetime and preexp fields no longer read
            ipx_header_expected = {'board_temp': 0.0, 'camera': 'Thermosensorik CMT 256 SM HS', 'ccd_temp': 59.0,
                                   'codex': 'JP1', 'date_time': '2013-10-23T15:37:29Z', 'depth': 14, 'exposure': 50.0,
                                   'file_format': 'IPX-2', 'filter': '', 'gain': np.array([0., 0.]), 'hbin': 0,
                                   'height': 32, 'is_color': 0, 'left': 0, 'lens': '', 'n_frames': 625,
                                   'offset': np.array([0., 0.]), 'orientation': 0, 'pre_exp': 0.0, 'shot': 30378,
                                   'strobe': 0, 'taps': 0, 'top': 153, 'trigger': -0.5, 'vbin': 0,
                                   'view': 'HL01 Upper divertor view#1', 'width': 256, 'bottom': 121, 'right': 256}

            ipx_meta_data['ipx_header'].pop('frame_times')
            # Check keys are identical
            self.assertEqual(sorted(ipx_meta_data['ipx_header']), sorted(ipx_header_expected))
            # Check values match
            self.assertTrue(np.all([np.all(ipx_meta_data['ipx_header'][key] == ipx_header_expected[key]) for key in
                                    ipx_header_expected]))

        def test_read_movie_data_uda_rit030378(self):
            # Ipx 2 file
            pulse = 30378
            camera = 'rit'

            # Read single frame
            n_start, n_end = 78, 78
            frames = [78]
            nframes = 1
            frame_nos, frame_times, frame_data = read_movie_data(pulse, camera, n_start=n_start, n_end=n_end)
            self.assertTrue(isinstance(frame_data, np.ndarray))
            self.assertTrue(isinstance(frame_nos, np.ndarray))
            self.assertTrue(isinstance(frame_times, np.ndarray))
            self.assertEqual(frame_data.shape, (nframes, 32, 256))
            self.assertEqual(frame_times.shape, (nframes,))
            self.assertEqual(frame_nos.shape, (nframes,))
            np.testing.assert_array_equal(frame_nos, frames)
            np.testing.assert_array_equal(frame_times, [0.04365])
            frame_data_expected = np.array([[6283., 6389.], [6253., 6426.], [6232., 6564.], [6248., 6399.]])
            np.testing.assert_array_equal(frame_data[0, ::10, ::150], frame_data_expected)

            # TODO test transforms

    # TODO: Add test for ipx2 frame data

    class TestIoUdaSlow(unittest.TestCase):

        def setUp(self):
            self.maxDiff = None

        def test_read_movie_data_uda_rir030378(self):
            # Ipx 1 file
            pulse = 30378
            camera = 'rir'

            # Read whole movie
            frame_nos, frame_times, frame_data = read_movie_data(pulse, camera)
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
            frame_nos, frame_times, frame_data = read_movie_data(pulse, camera, n_start=n_start, n_end=n_end)
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

        def test_read_movie_data_uda_rit030378(self):
            # Ipx 2 file
            pulse = 30378
            camera = 'rit'

            # Read whole movie
            frame_nos, frame_times, frame_data = read_movie_data(pulse, camera)
            self.assertTrue(isinstance(frame_data, np.ndarray))
            self.assertTrue(isinstance(frame_nos, np.ndarray))
            self.assertTrue(isinstance(frame_times, np.ndarray))
            # self.assertEqual(frame_data.shape, (3750, 8, 320))
            self.assertEqual(frame_data.shape, (625, 32, 256))
            self.assertEqual(frame_times.shape, (625,))
            self.assertEqual(frame_nos.shape, (625,))
            np.testing.assert_array_equal(frame_nos[[0, -1]], (0, 624))
            np.testing.assert_allclose(frame_times[[0, -1]], (-0.049949,  0.69885))
            frame_data_expected = np.array([[[6296., 6401.], [6313., 6431.], [6252., 6412.]],
                                            [[6307., 6411.], [6320., 6436.], [6260., 6420.]]])
            np.testing.assert_array_equal(frame_data[[295, 597], ::15, ::150], frame_data_expected)

            # Read frame range
            n_start, n_end = 100, 110
            frames = np.arange(n_start, n_end + 1)
            nframes = (n_end - n_start) + 1
            frame_nos, frame_times, frame_data = read_movie_data(pulse, camera, n_start=n_start, n_end=n_end)
            self.assertTrue(isinstance(frame_data, np.ndarray))
            self.assertTrue(isinstance(frame_nos, np.ndarray))
            self.assertTrue(isinstance(frame_times, np.ndarray))
            self.assertEqual(frame_data.shape, (nframes, 32, 256))
            self.assertEqual(frame_times.shape, (nframes,))
            self.assertEqual(frame_nos.shape, (nframes,))
            np.testing.assert_array_equal(frame_nos, frames)
            frame_times_expected = [0.07005, 0.07125, 0.07245, 0.07365, 0.07485, 0.07605, 0.07725,
                                    0.07845, 0.07965, 0.08085, 0.08205]
            np.testing.assert_allclose(frame_times, frame_times_expected)
            frame_data_expected = np.array([[[6288., 6389.], [6253., 6429.], [6235., 6565.], [6247., 6399.]],
                                            [[6288., 6390.], [6254., 6429.], [6232., 6566.], [6246., 6399.]]])
            np.testing.assert_array_equal(frame_data[[1, 3], ::10, ::150], frame_data_expected)

def suite():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    suite.addTests(loader.loadTestsFromTestCase(TestIoUdaFast))
    suite.addTests(loader.loadTestsFromTestCase(TestIoUdaSlow))
    return suite

if __name__ == '__main__':

    runner = unittest.TextTestRunner(failfast=True)
    # unittest.main()
    runner.run(suite())


