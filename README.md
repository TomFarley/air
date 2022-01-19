FIRE 
====
[![pipeline status](https://git.ccfe.ac.uk/MAST-U_Scheduler/air/badges/ci/pipeline.svg)](https://git.ccfe.ac.uk/MAST-U_Scheduler/air/commits/ci)
[![coverage report](https://git.ccfe.ac.uk/MAST-U_Scheduler/air/badges/ci/coverage.svg)](https://git.ccfe.ac.uk/MAST-U_Scheduler/air/commits/ci)

Analysis code called by the MAST-U and JET schedulers after each discharge to process infra-red camera data.

Documentation
--------------
If you are on the ukaea network, the full FIRE documentation can be found
[here](https://git.ccfe.ac.uk/MAST-U_Scheduler/air.gitlab.io/fire/docs/sphinx/build/html) at:
[https://git.ccfe.ac.uk/MAST-U_Scheduler/air.gitlab.io/fire/](https://git.ccfe.ac.uk/MAST-U_Scheduler/air.gitlab.io/fire/)

Alternatively open [``docs/sphinx/build/html/index.html``](docs/sphinx/build/html/index.html) from this repository.

Gitlab pages path should be: https://mast-u_scheduler.gitpages.ccfe.ac.uk/air?

For authorship information see AUTHORS.txt, and for details of how to cite the code in academic publications see CITE.txt.

Installing FIRE
---------------
It is recommended to create a virtual environment for FIRE to isolate it's
package dependencies from the rest of your system, while using the system
packages as a base:
```bash
# See below for installing venv packages if you don't already have them (pip show virtualenv)
$ mkvirtualenv fire --system-site-packages  # venv named 'fire'
$ workon fire  # This should put (fire) at start of terminal prompt. Use deactivate to revert to central python enviroment
```
Install fire in developer mode along with it's pypy dependencies :
```bash
$ cd <path_to_air_reposintory>  # location of git clone
$ python -m pip install -r requirements.txt
$ pip install --user -e .
```
The following ukaea gitlab require ukaea credentials and are most easily installed from cloned repositories (assuming [ssh keys](https://docs.gitlab.com/ee/ssh/#generate-an-ssh-key-pair) already set up).
```bash
$ cd <path_to_clone_repos_to>
$ git clone git@git.ccfe.ac.uk:MAST-U_Scheduler/air_calib.git  # Calibration data
# Optional dependecies:
$ git clone git@git.ccfe.ac.uk:MAST-U/mastvideo.git  # Required for reading local IPX files
$ git clone git@git.ccfe.ac.uk:tfarley/ir_analysis.git  # Scripts for performing analysis with fire
$ git clone git@git.ccfe.ac.uk:jrh/mastu_exhaust_analysis.git  # Used for efit equilibria amongst other things
$ git clone git@git.ccfe.ac.uk:SOL_Transport/pyEquilibrium.git # Used for efit equilibria
$ git clone git@git.ccfe.ac.uk:SOL_Transport/pyIpx.git  # Older alternative to mastvideo library

$ cd ..
$ pip install -e ir_tools mastvideo pyIpx ir_analysis mastu_exhaust_analysis

```
Run tests to confirm installation is successful:
```bash
$ python setup.py test
```

#### Installing venv packages
If you don't already have virtualenv and virtualenvwrapper installed follow the steps below before starting the installation process described above
```bash
$ module unload python
$ module load python/3.7  # Use python3.7 as base
$ pip install --user virtualenv virtualenvwrapper
$ export WORKON_HOME=~/Envs
$ source /usr/local/bin/virtualenvwrapper.sh  # Add this to .bashrc for future
```

## Configuring FIRE
Your user FIRE configuration is set up by default to be `~/.fire/.fire_config.json`
Instruction for configuring this coming soon...

Summary for running on the scheduler
------------------------------------

* Languge: Python 3.6+
* Scheduler signal dependencies:
    - Hard: None
    - Soft: Efit (in future)
* Command(s) for running the code (from the air repository directory):
    - `$ python fire/scripts/run_scheduler_workflow.py <camera_tag> <shot_number>, -pass <pass_number>`
    - See `$ python fire/scripts/run_fire.py --help` for the full call signature
    - No scheudler flag is currently requireed in the call
* Freia module dependencies:
    - FUN
    - python/3.7
    - uda-mast/
* Emails for recipients of automatic emails:
    - tom.farley@ukaea.uk
* Contact for advice using FIRE code:
    - Tom Farley, tom.farley@ukaea.uk (RO for IR cameras)