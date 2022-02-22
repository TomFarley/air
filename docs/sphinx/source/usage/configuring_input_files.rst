=======================
Configuring input files
=======================

This section discusses configuring the FIRE input files that can be found in the `air_calib <https://git.ccfe.ac
.uk/MAST-U_Scheduler/air_calib/>`_  gitlab repository.

FIRE has been written so as extract as much of the settings and configuration from the code as possible. This means
that the main `air` repository code base should generally only need modification if new features are added to the code.
All configuration of how the code is run can be set either by modifying calibration or input files in the `air_calib`
 repository or in user configuration files in the user's home directory.
This should lead to a clean git revision history for `air` where a commit indicates a feature modification or bug fix
 rather than a minor configuration change.

User FIRE configuration
-----------------------
Personalisation of the fire configuration is handled through a user configuration file. This file should be located
in the users `fire_user_dir` (see :ref:`path-conventions`)
When FIRE is run for the first time, if it doesn't find user settings in the default location (`~/fire/`) it will default to creating this directory and populating it with a default user configuration file `fire_config.json`.
If you would like your fire user directory (location for user settings and output figures and files etc) then your alternative path can either be passed to schduler_workflow(user_path=...) each time or it can be set more permanently by settings the `FIRE_USER_DIR` enviroment variable in your

.bashrc:.. code-block:: bash

    export FIRE_USER_DIR=<my_fire_user_directory_path>


.. code-block:: shell

    $ fire <pulse_no> <camera> <pass> <machine> <schedluer> <magnetics>