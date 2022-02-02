FIRE 
====
[![pipeline status](https://git.ccfe.ac.uk/MAST-U_Scheduler/air/badges/ci/pipeline.svg)](https://git.ccfe.ac.uk/MAST-U_Scheduler/air/commits/ci)
[![coverage report](https://git.ccfe.ac.uk/MAST-U_Scheduler/air/badges/ci/coverage.svg)](https://git.ccfe.ac.uk/MAST-U_Scheduler/air/commits/ci)

FIRE (Fusion Infra Red Experiments) is the analysis code called by the MAST-U schedulers after each discharge to process infra-red camera data.

Installing FIRE
---------------
It is recommended to create a virtual environment for FIRE to isolate it's
package dependencies from the rest of your system, while using the system
packages as a base (for instructions on installing the virtualenv and virtualenvwrapper packages see the [section below](#installing-venv-packages)):
```bash
$ mkvirtualenv fire --system-site-packages  # venv named 'fire'
$ workon fire  # This should put (fire) at start of terminal prompt. Use deactivate to revert to central python enviroment
```
Install fire in developer mode along with it's pypy dependencies :
```bash
$ cd <path_to_air_reposintory>  # location of git clone
$ python -m pip install -r requirements.txt
$ pip install --user -e .
```
The following ukaea gitlab require ukaea credentials and are most easily installed from cloning and pip installing the repositories or pip installing them directory from the urls as shown below:
```bash
$ pip install git+https://github.com:euratom-software/calcam.git  # Spatial calibration classes (should be installed by by setup.py?)
$ pip install git+https://git.ccfe.ac.uk:jrh/mastu_exhaust_analysis.git  # Used for efit equilibria amongst other things
$ pip install git+https://git.ccfe.ac.uk:MAST-U/mastvideo.git  # Required for reading local IPX files
$ pip install git+https://git.ccfe.ac.uk:SOL_Transport/pyEquilibrium.git # Used for efit equilibria
# Optional dependecies:
$ pip install git+https://git.ccfe.ac.uk:SOL_Transport/pyIpx.git  # Older alternative to mastvideo library
```

Several other ukaea repos also complement workign with FIRE and are also recommended. These can be cloned and installed as shown bellow (assuming [ssh keys](https://docs.gitlab.com/ee/ssh/#generate-an-ssh-key-pair) already set up)
```bash
$ cd <directory_to_clone_repos>
$ git clone git@git.ccfe.ac.uk:MAST-U_Scheduler/air_calib.git  # Calibration data
$ git clone git@git.ccfe.ac.uk:tfarley/ir_analysis.git  # Scripts for performing analysis runs with FIRE
$ pip install git+https://git.ccfe.ac.uk:tfarley/ir_tools.git  # Scripts for working with IR data, producing calcam calibration images etc

$ cd ..
$ pip install -e ir_tools ir_analysis

```
Run tests to confirm installation is successful:
```bash
$ python setup.py test
```

#### Installing venv packages
If you don't already have virtualenv and virtualenvwrapper installed (check with `pip show virtualenv`) follow the steps below before starting the installation process described above. You can also work with a plain virtualenv if you would prefer not to use virtualenvwrapper.
```bash
$ module unload python
$ module load python/3.7  # Use python3.7 as base
$ pip install --user virtualenv virtualenvwrapper
$ export WORKON_HOME=~/Envs
$ source /usr/local/bin/virtualenvwrapper.sh  # Add this to .bashrc for future
```

## Configuring FIRE
When FIRE is run for the first time, if it doesn't find user settings in the default location (`~/fire/`) it will default to creating this directory and populating it with a default user configuration file `fire_config.json`.
If you would like your fire user directory (location for user settings and output figures and files etc) then your alternative path can either be passed to schduler_workflow(user_path=...) each time or it can be set more permanently by settings the `FIRE_USER_DIR` enviroment variable in your .bashrc:
```bash
export FIRE_USER_DIR=<my_fire_user_directory_path>
```

Instruction for configuring your `fire_config.json` coming soon...

Summary for running the scheduler code
--------------------------------------

* Languge: Python 3.6+
* Scheduler signal dependencies:
    - Hard: None
    - Soft: Efit (in future)
* Command(s) for running the code (from the air repository directory):
    - `$ python fire/scripts/run_fire.py <camera_tag> <shot_number>, -pass <pass_number>`
        - eg `$ python fire/scripts/run_fire.py rit 44628`
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

Documentation
--------------
If you are on the ukaea network, the full FIRE documentation can be found
[here](https://git.ccfe.ac.uk/MAST-U_Scheduler/air.gitlab.io/fire/docs/sphinx/build/html) at:
[https://git.ccfe.ac.uk/MAST-U_Scheduler/air.gitlab.io/fire/](https://git.ccfe.ac.uk/MAST-U_Scheduler/air.gitlab.io/fire/)

Alternatively open [``docs/sphinx/build/html/index.html``](docs/sphinx/build/html/index.html) from this repository.

Gitlab pages path should be: https://mast-u_scheduler.gitpages.ccfe.ac.uk/air?

For authorship information see AUTHORS.txt, and for details of how to cite the code in academic publications see CITE.txt.