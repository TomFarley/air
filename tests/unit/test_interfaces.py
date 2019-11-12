import pytest
import unittest
from pathlib import Path

import numpy as np

from fire.interfaces.interfaces import (json_dump, json_load, read_movie_meta_data,
                    generate_pulse_id_strings, generate_camera_id_strings, generate_frame_id_strings)
from fire import fire_paths

pwd = Path(__file__).parent

class TestInterfaces(unittest.TestCase):

    def setUp(self):
        path_fn = fire_paths['root'] / 'input_files/user/fire_config.json.tmp'
        self.json_path_fn = path_fn
        if path_fn.exists():
            path_fn.unlink()

    def test_json_dump(self):
        obj = {"user_details":
            {"name": None, "email": None},
              "default_params":
                  {"machine": "mast", "pulse": 23586, "camera": "rir", "pass": 0},
              "paths":
                  {"movie_plugins": ["{fire_path}/interfaces/movie_plugins/"],
                    "machine_plugins": ["{fire_path}/interfaces/machine_plugins/"],
                    "input_file_types": ["pulse_lookups", "black_body_calibrations", "surface_properties"],
                    "input_files": ["~/fire/input_files/{machine}/",
                                    "{fire_path}/input_files/{machine}/"],
                    "calcam_calibrations": ["~/calcam/calibrations/",
                                            "~/calcam2/calibrations/"]
                  },
        }
        path_fn = self.json_path_fn
        path_fn = json_dump(obj, path_fn)
        self.assertTrue(path_fn.exists())

        with self.assertRaises(FileExistsError):
            path_fn = json_dump(obj, path_fn.parent, path_fn.name, overwrite=False)

        obj = {('my', 'tuple', 'key'): 'value'}
        with self.assertRaises(TypeError):
            path_fn = json_dump(obj, path_fn)

    def test_json_load(self):
        path_fn = fire_paths['root'] / 'input_files/user/fire_config.json'

        config = json_load(path_fn)
        self.assertTrue(isinstance(config, dict))
        self.assertTrue(len(config), 6)
        self.assertEqual(config['default_params']['pass'], 0)
        config = None

        config = json_load(str(path_fn.parent), fn=path_fn.name)
        self.assertTrue(len(config), 6)
        self.assertEqual(config['default_params']['pass'], 0)

        out = json_load(path_fn, keys=['default_params', 'pass'])
        self.assertEqual(out, 0)

        with self.assertRaises(KeyError):
            out = json_load(path_fn, keys=['default_params', 'my_made_up_key'])

        with self.assertRaises(FileNotFoundError):
            out = json_load('my_path')


    def test_read_movie_meta_data(self):
        user_defaults = {'machine': 'MAST', 'camera': 'rit', 'pulse': 23586}


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


    def tearDown(self):
        path_fn = self.json_path_fn
        if path_fn.exists():
            path_fn.unlink()


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