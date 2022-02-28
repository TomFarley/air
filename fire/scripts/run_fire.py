#!/usr/bin/env python

"""
Submission script for MAST-U/JET IR scheduler analysis code.

Call signature from command line:
$ python run_fire.py <camera_tag> <pulse>
For help output run:
$ python run_fire.py -h

Usage example:
$ python run_fire.py rit 44805
$ python run_fire.py rit 44805 -pass 2 -machine mast_u -equilibrium -checkpoints

Created: 10-10-2019
"""

import argparse
import pathlib

MACHINE_DEFAULT = 'MAST_U'
N_CORES_DEFAULT = 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Fusion Infra Red Experiments scheduler code')
    parser.add_argument("camera", help="Raw tag name of camera to analyse e.g. 'rir'", type=str)
    parser.add_argument("pulse", help="Shot/pulse number to analyse e.g. 50000", type=int, metavar='shot')
    parser.add_argument("-p", "--pass", help="Pass number of analysis", type=int, default=0, metavar='PASS_NO',
                        dest='pass_no')
    parser.add_argument("-machine", help="Name of machine (tokamak) being analysed", type=str,
                        default=MACHINE_DEFAULT, metavar='TOKAMAK_NAME')
    parser.add_argument("-a", "--alpha",
                        help="User value for THEODOR alpha parameter to override value set in material properties file",
                        dest="alpha_user", default=None)
    parser.add_argument("-e", "--equilibrium", help="Whether to run analysis with efit as a dependency",
                        action="store_true", default=False)
    parser.add_argument("-c", "--checkpoints", help="Whether to reproduce cached data files",
                        action="store_true", default=False, dest='update_checkpoints')
    parser.add_argument("-n", "--n_cores", help="Number of cores to use for multiprocessing", type=int,
                        default=N_CORES_DEFAULT)
    parser.add_argument("-u", "--path_user", help="Path to users fire directory containing fire_config.json",
                        type=pathlib.Path,
                        default=None)  # If None use value from user's environment variable $FIRE_USER_DIR else ~/fire
    parser.add_argument("-o", "--path_out", help="Directory path to write output UDA netcdf file to", type=pathlib.Path,
                        default=None)  # If None will use value from user's fire_config.json file
    parser.add_argument("-i", "--path_calib", help="Directory path to retrieve input calibration files from",
                        type=pathlib.Path, default=None)  # If None will use value from user's fire_config.json file
    parser.add_argument("-f", "--movie_plugins_filter",
                        help="Which movie plugin formats to use to attempt to read movie data",
                        type=pathlib.Path, default=None)  # If None will use value from user's fire_config.json file
    parser.add_argument("-s", "--scheduler", help="Whether the analysis is being run of the scheduler machine",
                        action="store_true", default=True)

    args = vars(parser.parse_args())
    print(f'run_fire.py called with args: {args}')

    from fire.scripts.scheduler_workflow import scheduler_workflow
    scheduler_workflow(**args)
