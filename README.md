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
$ cd <path_to_user_virtualenvs>  # eg. ~/venv or ~/.venv
$ python -m venv --system-site-packages fire  # venv named 'fire'
$ source fire/bin/activate  # use command 'deactivate' to unset
```
Install fire and it's dependencies in developer mode:
```bash
$ cd <path_to_air_reposintory>  # location of git clone
$ python -m pip install -r requirements.txt
$ pip install --user -e .
```
Run tests to confirm installation is successful:
```bash
$ python setup.py test
```

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