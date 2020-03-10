============
Installation
============

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
Make sure the CAD models used in your fire config file (~/.fire_config.json) are known to calcam.
