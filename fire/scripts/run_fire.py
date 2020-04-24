#!/usr/bin/env python

"""
Submission script for MAST-U/JET IR scheduler analysis code.

Call signature from command line:
$ python run_fire <pulse> <pass> <machine>

Usage example:
$ python run_fire 50100 0 MASTU

Created: 10-10-2019
"""

import argparse

MACHINE_DEFAULT = 'MAST_U'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Fusion Infra Red Experiments scheduler code')
    parser.add_argument("camera", help="Tag name of camera to analyse e.g. 'rir'", type=str)
    parser.add_argument("pulse", help="Shot/pulse number to analyse e.g. 50000", type=int, metavar='shot')
    parser.add_argument("-pass", help="Pass number of analysis", type=int, default=0, metavar='PASS_NO', dest='pass_no')
    parser.add_argument("-machine", help="Name of machine (tokamak) being analysed", type=str,
                        default=MACHINE_DEFAULT, metavar='TOKAMAK_NAME')
    parser.add_argument("-equilibrium", help="Whether to run analysis with efit as a dependency", action="store_true",
                        default=False)
    parser.add_argument("-checkpoints", help="Whether to reproduce cached data files", action="store_true",
                        default=False, dest='update_checkpoints')

    args = vars(parser.parse_args())
    print(args)
    # camera = args.pop('camera')
    # camera = args.pop('shot')
    from fire.scripts.scheduler_workflow import scheduler_workflow
    scheduler_workflow(**args)
