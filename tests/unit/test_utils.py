import pytest
import unittest
from pathlib import Path

import numpy as np

from fire.utils import update_call_args

pwd = Path(__file__).parent

class TestUtils(unittest.TestCase):

    def test_update_call_args(self):
        user_defaults = {'pulse': 30378, 'camera': 'rir', 'machine': 'MAST_U'}

        inputs = (29852, 'rir', 'MAST')
        outputs = update_call_args(user_defaults, *inputs)
        expected = inputs
        self.assertEqual(outputs, expected)

        inputs = (None, 'rir', 'MAST')
        outputs = update_call_args(user_defaults, *inputs)
        expected = (30378, 'rir', 'MAST')
        self.assertEqual(outputs, expected)

        inputs = (None, 'rit', None)
        outputs = update_call_args(user_defaults, *inputs)
        expected = (30378, 'rit', 'MAST_U')
        self.assertEqual(outputs, expected)

def suite():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    # suite.addTests(loader.loadTestsFromTestCase(TestIoIpx))
    suite.addTest(TestUtils.test_update_call_args)
    return suite

if __name__ == '__main__':

    runner = unittest.TextTestRunner(failfast=True)
    # unittest.main()
    runner.run(suite())