#!/usr/bin/env python

"""


Created: 
"""

import unittest
from pathlib import Path

import numpy as np

from fire.interfaces.user_config import get_user_fire_config, _read_user_fire_config

class TestUserConfig(unittest.TestCase):

    def setUp(self):
        pass

    def test_get_user_fire_config(self):
        path = None
        fn = None

        fire_config_raw = _read_user_fire_config(path=path, fn=fn)
        path_raw = str(fire_config_raw['user']['paths']['calibration_files'][0])

        fire_config_updated, config_groups, config_path_fn = get_user_fire_config(path=path, fn=fn)
        path_updated = str(fire_config_updated['user']['paths']['calibration_files'][0])

        self.assertTrue('{' in path_raw)
        self.assertTrue('{' not in path_updated)
        self.assertTrue('fire_paths' in config_groups)

def suite():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    suite.addTests(loader.loadTestsFromTestCase(TestUserConfig))
    # suite.addTest(TestUserConfig.)
    return suite

if __name__ == '__main__':

    runner = unittest.TextTestRunner(failfast=True)
    unittest.main()
    # runner.run(suite())