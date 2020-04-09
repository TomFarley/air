import unittest
from pathlib import Path

import calcam

from fire.scripts.scheduler_workflow import scheduler_workflow

pwd = Path(__file__).parent

class TestSchedulerWorkflow(unittest.TestCase):

    def setUp(self):
        conf = calcam.config.CalcamConfig()
        models = conf.get_cadmodels()
        print(models)
        if 'MAST' not in models:
            cad_path = (pwd / '../../fire/input_files/cad/').resolve()
            print(f'Add CAD path: {cad_path}')
            conf.cad_def_paths.append(str(cad_path))
            conf.save()
            print(models)
            # self.assertTrue('MAST' in models)

    def test_scheduler_workflow(self):
        inputs = {'pulse': 23586, 'camera': 'rir', 'machine': 'MAST', 'pass_no': 0, 'scheduler': False,
                  'magnetics': False, 'update_checkpoints': True,
                  'debug': {'raycast': False}, 'figures': {'spatial_res': False}
        }
        scheduler_workflow(**inputs)
        # TODO: test output

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