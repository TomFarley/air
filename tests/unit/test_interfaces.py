import pytest
import unittest
from pathlib import Path

import numpy as np

from fire.interfaces.interfaces import (load_user_defaults, read_movie_meta_data,
                    generate_pulse_id_strings, generate_camera_id_strings, generate_frame_id_strings)

pwd = Path(__file__).parent

class TestInterfaces(unittest.TestCase):

    def test_load_user_defaults(self):
        user_defaults = {'machine': 'MAST', 'camera': 'rit', 'pulse': 23586}

        out = load_user_defaults()
        self.assertEqual(out, user_defaults)

    def test_read_movie_meta_data(self):
        user_defaults = {'machine': 'MAST', 'camera': 'rit', 'pulse': 23586}

        out = load_user_defaults()
        self.assertEqual(out, user_defaults)

    def test_id_strings(self):
        values = {'machine': 'MAST', 'camera': 'rit', 'pass_no': 1, 'pulse': 23586, 'lens': '25mm', 't_int': 2.7e-5,
                  'frame_no': 1234, 'frame_time': 0.5123}

        id_strings = {}
        kwargs = {key: values[key] for key in ['pulse', 'camera', 'machine', 'pass_no']}
        id_strings = generate_pulse_id_strings(id_strings, **kwargs)
        expected = {'pulse_id': 'MAST-23586', 'camera_id': 'MAST-23586-rit', 'pass_id': 'MAST-23586-1'}
        self.assertDictEqual(id_strings, expected)

        kwargs = {key: values[key] for key in ['lens', 't_int']}
        id_strings = generate_camera_id_strings(id_strings, **kwargs)
        expected = {'pulse_id': 'MAST-23586', 'camera_id': 'MAST-23586-rit', 'pass_id': 'MAST-23586-1',
                    'lens_id': 'MAST-23586-rit-25mm', 't_int_id': 'MAST-23586-rit-25mm-2.7e-05'}
        self.assertDictEqual(id_strings, expected)

        kwargs = {key: values[key] for key in ['frame_no', 'frame_time']}
        id_strings = generate_frame_id_strings(id_strings, **kwargs)
        expected = {'pulse_id': 'MAST-23586', 'camera_id': 'MAST-23586-rit', 'pass_id': 'MAST-23586-1',
                    'lens_id': 'MAST-23586-rit-25mm', 't_int_id': 'MAST-23586-rit-25mm-2.7e-05',
                    'frame_id': 'MAST-23586-rit-1234', 'time_id': 'MAST-23586-0.5123'}
        self.assertDictEqual(id_strings, expected)


def suite():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    # suite.addTests(loader.loadTestsFromTestCase(TestIoIpx))
    suite.addTest(TestInterfaces.test_update_call_args)
    return suite

if __name__ == '__main__':

    runner = unittest.TextTestRunner(failfast=True)
    unittest.main()
    # runner.run(suite())