import pytest
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from fire.utils import update_call_args, locate_file, make_iterable, movie_data_to_xarray

pwd = Path(__file__).parent

class TestUtils(unittest.TestCase):

    def test_movie_data_to_xarray(self):
        n_frames = 20
        frame_shape = (256, 320)
        frame_times = np.linspace(0, 0.15, n_frames)
        frame_nos = np.arange(115, 115+n_frames)
        frame_data = np.ones((n_frames, *frame_shape))

        frame_data = movie_data_to_xarray(frame_data, frame_times, frame_nos)
        self.assertTrue(isinstance(frame_data, xr.DataArray))
        np.testing.assert_array_equal(frame_data['t'].values, frame_times)
        np.testing.assert_array_equal(frame_data['n'].values, frame_nos)
        self.assertEqual(frame_data['t'].attrs['units'], 's')

    def test_update_call_args(self):
        user_defaults = {'pulse': 30378, 'camera': 'rir', 'machine': 'MAST_U'}

        inputs = (29852, 'rir', 'MAST')
        outputs = update_call_args(user_defaults, *inputs)
        expected = (29852, 'rir', 'mast')
        self.assertEqual(outputs, expected)

        inputs = (None, 'rir', 'MAST')
        outputs = update_call_args(user_defaults, *inputs)
        expected = (30378, 'rir', 'mast')
        self.assertEqual(outputs, expected)

        inputs = (None, 'rit', None)
        outputs = update_call_args(user_defaults, *inputs)
        expected = (30378, 'rit', 'mast_u')
        self.assertEqual(outputs, expected)

        inputs = (None, None, None)
        outputs = update_call_args(user_defaults, *inputs)
        expected = (30378, 'rir', 'mast_u')
        self.assertEqual(outputs, expected)

    def test_locate_file(self):
        file = Path(__file__)
        paths = ['/some/{special}/path/', file.parent, '~']
        fns = ['incorrect_name.ext', file.name, '{var}_name.txt']
        path_kws = {'special': 'wonderful'}
        fn_kws = {'var': 'my'}

        p, f = locate_file(paths, fns, path_kws=path_kws, fn_kws=fn_kws, return_raw_fn=False, return_raw_path=False)
        self.assertEqual(p, paths[1])
        self.assertEqual(f, fns[1])

        # TODO: Test returning raw for subbed args

    def test_make_iterable(self):
        # Scalars - nest
        input = 3
        output = make_iterable(input, ndarray=False, cast_to=None, cast_dict=None, nest_types=None)
        self.assertEqual(output, [input])
        input = 'hello'
        output = make_iterable(input, ndarray=False, cast_to=None, cast_dict=None, nest_types=None)
        self.assertEqual(output, [input])
        input = 43653.547
        output = make_iterable(input, ndarray=False, cast_to=None, cast_dict=None, nest_types=None)
        self.assertEqual(output, [input])
        input = None
        output = make_iterable(input, ndarray=False, cast_to=None, cast_dict=None, nest_types=None)
        self.assertEqual(output, [input])
        input = Path('~/path')
        output = make_iterable(input, ndarray=False, cast_to=None, cast_dict=None, nest_types=None)
        self.assertEqual(output, [input])

        # Already iterables
        input = (1, 4.6, 'str')
        output = make_iterable(input, ndarray=False, cast_to=None, cast_dict=None, nest_types=None)
        self.assertEqual(output, input)
        input = np.linspace(0, 10, 11)
        output = make_iterable(input, ndarray=False, cast_to=None, cast_dict=None, nest_types=None)
        np.testing.assert_equal(output, input)
        input = ['hello', 'bye']
        output = make_iterable(input, ndarray=False, cast_to=None, cast_dict=None, nest_types=None)
        self.assertEqual(output, input)
        input = {'hello': 'bye'}
        output = make_iterable(input, ndarray=False, cast_to=None, cast_dict=None, nest_types=None)
        self.assertEqual(output, input)

        # To ndarray
        input = [1, 7, 8.5, 13, 2]
        output = make_iterable(input, ndarray=True, cast_to=None, cast_dict=None, nest_types=None)
        np.testing.assert_equal(output, np.array(input))
        input = 3
        output = make_iterable(input, ndarray=True, cast_to=None, cast_dict=None, nest_types=None)
        np.testing.assert_equal(output, np.array([input]))

        # cast_to
        input = [1, 7, 8.5, 13, 2]
        output = make_iterable(input, ndarray=False, cast_to=tuple, cast_dict=None, nest_types=None)
        self.assertEqual(output, tuple(input))

        # cast_dict
        input = [1, 7, 8.5, 13, 2]
        output = make_iterable(input, ndarray=False, cast_to=None, cast_dict={list:tuple}, nest_types=None)
        self.assertEqual(output, tuple(input))

        # nest_types
        input = {'hello': 'bye'}
        output = make_iterable(input, ndarray=False, cast_to=None, cast_dict=None, nest_types=(dict,))
        self.assertEqual(output, [input])


def suite():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    # suite.addTests(loader.loadTestsFromTestCase(TestIoIpx))
    suite.addTest(TestUtils.test_update_call_args)
    return suite

if __name__ == '__main__':

    runner = unittest.TextTestRunner(failfast=True)
    unittest.main()
    # runner.run(suite())