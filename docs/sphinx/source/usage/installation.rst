============
Installation
============

Installing FIRE
---------------

It is recommended that these instructions are followed using the bash shell. Make sure you have seeral GB of disk quota available for a smooth installation. Clone the air repository in a location of your choice:
.. code-block:: shell
    $ cd <path_to_clone_air_reposintory>  # location to clone air repo to
    $ cd air
    $ git checkout dev  # Switch to the up to date dev branch

It is recommended to create a [virtual environment](https://realpython.com/python-virtual-environments-a-primer/#using-virtual-environments) for FIRE to isolate it's
package dependencies from the rest of your system, while using the system
packages as a base (for instructions using virtualenvwrapper see the [section below](#installing-venv-packages)). To clone FIRE and set up a virtual enviroment on freia:
.. code-block:: shell
     # From air/ directory

    $ module switch python/3.7  # Use python3.7 as base for your venv
    $ pip install --user virtualenv  # If not already installed
    $ python3 -m venv env --system-site-packages --upgrade-deps  # build new venv from central python installation

    # Every time you work with the venv you will need to do:
    $ source env/bin/activate  # Activate the virtual enviroment

Now your shell prompt should be proceeded by `(env) $` indicating your venv is active. Now any python packages you install will be installed in your air/venv directory and the `--user` argument to pip is no longer necessary. Use the `deactivate` command to revert to central python environment.

Install fire in developer mode along with it's pypy dependencies :
```bash
 # From air/ directory

$ pip install --user -e .
```
The following ukaea gitlab repositories require ukaea credentials and so are most easily installed from cloned repositories (assuming [ssh keys](https://docs.gitlab.com/ee/ssh/#generate-an-ssh-key-pair) already set up):
```bash
$ cd <path_to_clone_reposintories>  # location to clone repos

$ git clone git@github.com:euratom-software/calcam.git  # Spatial calibration classes.
 # If you don't have ssh keys set up with _github_ use https:
 #  $ git clone https://github.com/euratom-software/calcam.git

$ git clone git@git.ccfe.ac.uk:jrh/mastu_exhaust_analysis.git  # Used for efit equilibria amongst other things
$ git clone git@git.ccfe.ac.uk:MAST-U/mastvideo.git  # Required for reading local IPX files
$ git clone git@git.ccfe.ac.uk:SOL_Transport/pyEquilibrium.git # Used for efit equilibria
# Optional dependecies:
$ git clone git@git.ccfe.ac.uk:SOL_Transport/pyIpx.git  # Older alternative to mastvideo library

# Install the packages in developer mode
$ pip install -e calcam
$ pip install -e mastu_exhaust_analysis
$ pip install -e mastvideo
$ pip install -e pyEquilibrium pyIpx
```

Several other ukaea repos also complement working with FIRE and are also recommended. These can be cloned and installed as shown bellow
```bash
$ cd <directory_to_clone_repos>

$ git clone git@git.ccfe.ac.uk:MAST-U_Scheduler/air_calib.git  # Calibration data
$ git clone git@git.ccfe.ac.uk:mast-u-diagnostics/ir_analysis.git  # Scripts for performing analysis runs with FIRE, provenance capture etc.
$ git clone git@git.ccfe.ac.uk:mast-u-diagnostics/ir_tools.git# Scripts for working with IR data, producing calcam calibration images etc

$ pip install -e ir_tools
$ pip install -e ir_analysis
```
If working on Freia you may need to configure several settings. It is recommended these commands are added to your ~/.bashrc so you don't have to manually run them every time you want to work with FIRE. Alternatively you can source the example bashrc file in this repository (air/fire/input_files/user/bashrc_fire).
```bash
module purge
module load FUN
moudle swap python/3.7
module load vtk7/3.5.1  # Needed for Calcam renders to work
export FIRE_USER_DIR="<path_to_my_chosen_directory>"  # Only necessary if you don't want to use the default "~/fire" directory

# If using virtualenvwrapper (see below):
export WORKON_HOME=~/Envs
source ~/.local/bin/virtualenvwrapper.sh

```

Run tests to confirm installation of FIRE is successful:
```bash
$ pytest tests/test_suite_fast.py  # Fast

$ python setup.py test  # Slow

$ python fire/scripts/run_fire_example.py  # Example run
```

#### Using virtualenvwrapper packages
As an alternative to creating a virtualenv directory in the air repo it can be nice to work with virtualenvwrapper (check installed with `pip show virtualenv`) which place all your venvs in one directory and provides some convenience functions. To use this follow the steps below before starting the installation process described above.
```bash
$ module unload python
$ module load python/3.7  # Use python3.7 as base
$ pip install --user virtualenv virtualenvwrapper

 # Add these two lines to your .bashrc to save running them each time
$ export WORKON_HOME=~/Envs
$ source /usr/local/bin/virtualenvwrapper.sh

 # Create a venv named 'fire'
$ mkvirtualenv fire --system-site-packages

 # Use workon every time you want to work with this venv:
$ workon fire  # This should put (fire) at start of terminal prompt. Use deactivate to revert to central python enviroment
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

Installation Troubleshooting
----------------------------
- skimage ImportError
    - Details: Sometimes the installation results in a more recent version of scikit-image being installed which causes issues.
    - Solution:
        - With FIRE venv active:
        `$ pip install --upgrade scikit-image==0.18.3`
- "TypeError: load() missing 1 required positional argument: 'Loader'":
    - Details: Not sure if this is due to outdated dask or yaml packages?
    - Solution: Try each of these steps (with venv active):
        - `pip install --upgrade dask`
        - `pip install --upgrade pyyaml==6.0`
        - `pip install --upgrade distributed`
- Calcam CAD error
    - Solution: Configure Calcam with CAD location
        - `$ python`
        - `>>> import calcam`
        - `>>> calcam.start_gui()`
        - Click 'Settings' and add path to .ccm CAD files in e.g. air_calib/cad

See [BUGS_LOG.md](BUGS_LOG.md) for further information of package incompatibilities.

Cloning the FIRE repository
---------------------------

Make sure you have your ssh key setup on ccfe gitlab ([guide](https://git.ccfe.ac.uk/help/ssh/README#generating-a-new-ssh-key-pair)).
Then to recursively download the repository and its CCFE dependencies:

.. code-block:: shell

    $ git clone --recursive -j8 git@git.ccfe.ac.uk:MAST-U_Scheduler/air.git

Install FIRE with pip
---------------------
To install as a developer (so that the importable module tracks your local changes to the
code without reinstalling), from the top level ``FIRE`` folder containing ``setup.py`` run:

.. code-block:: shell

    $ cd <path to top FIRE directory>
    $ pip install --user -e .

If you will not be editing the code you can omit the ``-e`` option for a normal install.
If you are installing ``FIRE`` on a machine where you have admin priviledges you can also
omit ``--user``, so that the package is installed in your root python packages directory.

Point calcam to CAD models
--------------------------
In order to project lines of sight onto points surfaces in the cameras field of view, calcam needs to know the locations
of the appropriate calcam cad ".ccc" files.
To do this, launch the calcam gui, click "settings" and under CAD definitions click "Add" to direct calcam to the
folder(s) captaining the .ccc files.
Make sure the CAD models used in your fire config file (~/fire/fire_config.json) are known to calcam.
