import unittest
from pathlib import Path
import re, logging

import calcam

import fire
from fire.scripts.scheduler_workflow import scheduler_workflow

pwd = Path(__file__).parent

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class TestSchedulerWorkflowJet(unittest.TestCase):

    def setUp(self):
        self.fn_out = None
        conf = calcam.config.CalcamConfig()
        models = conf.get_cadmodels()
        logger.info('CAD models %s' % models)
        if 'MAST' not in models:
            cad_path = (pwd / '../../fire/input_files/cad/').resolve()
            logger.info(f'Add CAD path: {cad_path}')
            conf.cad_def_paths.append(str(cad_path))
            conf.save()
            logger.info('CAD models %s' % models)
            # self.assertTrue('MAST' in models)
        # TODO: Delete existing output files in repo dir '{diag_tag}{shot:06d}.nc'

    def test_scheduler_workflow_jet_kldt(self):
        pulses = [94935]
        inputs = {'pulse': 94935, 'camera': 'kldt', 'machine': 'JET', 'pass_no': 0, 'scheduler': False,
                  'equilibrium': False, 'update_checkpoints': True,
                  'debug': {'raycast': False}, 'figures': {'spatial_res': False}
        }
        path_out = Path(fire.fire_paths['root'] / '../').resolve()

        for pulse in pulses:
            inputs['pulse'] = pulse
            fn_out = '{camera}{pulse:06d}.nc'.format(**inputs)
            self.fn_out = path_out / re.sub("^r", 'a', str(fn_out))  # Change diagnostic tag from raw to analysed
            if self.fn_out.is_file():
                self.fn_out.unlink()

            self.assertFalse(self.fn_out .is_file())

            status = scheduler_workflow(**inputs)

            self.assertTrue(self.fn_out.is_file(), f'Failed to produce uda output file: {fn_out}')
            # TODO: test output file contents

    def tearDown(self):
        if (self.fn_out is not None) and self.fn_out.is_file():
            self.fn_out.unlink()

def suite():
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    # suite.addTests(loader.loadTestsFromTestCase(TestIoIpx))
    suite.addTest(TestSchedulerWorkflowJet.test_scheduler_workflow_jet_kldt)
    return suite

if __name__ == '__main__':

    runner = unittest.TextTestRunner(failfast=True)
    unittest.main()
    # runner.run(suite())