import pytest
import unittest
from pathlib import Path

import numpy as np

from fire.scheduler_workflow import scheduler_workflow

pwd = Path(__file__).parent

class TestSchedulerWorkflow(unittest.TestCase):

    def test_scheduler_workflow(self):
        inputs = {'pulse': 23586, 'camera': 'rir', 'machine': 'MAST', 'pass_no': 0, 'scheduler': False,
                  'magnetics': False}
        scheduler_workflow(**inputs)

def suite():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    # suite.addTests(loader.loadTestsFromTestCase(TestIoIpx))
    suite.addTest(TestSchedulerWorkflow.test_scheduler_workflow)
    return suite

if __name__ == '__main__':

    runner = unittest.TextTestRunner(failfast=True)
    unittest.main()
    # runner.run(suite())