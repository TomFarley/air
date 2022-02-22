Concepts and conventions
========================

.. _path-conventions:
Path conventions
----------------
FIRE makes use of a number of top level base paths that are substituted into path format strings in order to enable
easy customisation of input and output file locations.
Many of these paths are set with a longer explicit name in your fire_config.json file, but can be substituted into
path format strings (e.g. "{fire_user_dir}/figures/heatflux_maps/") using a concise name
The table below summarises these base paths:

.. list-table:: Input files
    :widths: 15 50 30 30
    :header-rows: 1

    * - Name
      - Description
      - Where set
      - Example/default
    * - {air_repo_dir}
      - Path to the air/ directory that is produced when you do git clone this repo
      - Wherever you clone the air repo
      - N/A
    * - {fire_source_dir}
      - Path `fire/` python source code directory inside the cloned `air/` directory
      - {air_repo_dir}/fire
      - N/A
    * - {fire_user_dir}
      - | Location for users input and output files.
        | Ideally not in the air source directory as it's contents is specific to the user.
        | * Contains the users fire_config.json file that specifies the values of other paths in this table.
      - Can be set with `export FIRE_USER_DIR=<my_fire_user_directory_path>`, othersise defaults to ~/fire/
      - ~/fire/
    * - {calib_dir}
      - Location of the cloned air_calib directory. If this is cloned from the same location the air repository was
        cloned, these files will be located using the default user_config.json settings
      - | Set in user config file: {fire_user_dir}/fire_config.json
        | Entry in file at: config['user']['paths']['calibration_files']
      - "{air_repo_dir}/../air_calib/"
    * - {figures_dir}
      - Location to save output figures eg from debugging etc (in nested directories)
      - | Set in user config file: {fire_user_dir}/fire_config.json
        | Entry in file at: config['user']['paths']['figures']
      - "{fire_user_dir}/figures/"