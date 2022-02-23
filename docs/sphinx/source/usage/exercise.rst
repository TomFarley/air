========
Exercise
========

This exercise has been written to help new users become familiar with performing new analyses with the FIRE code.
The exercise imagines that you are analysing data from a new IR camera for the first time and need to configure the
appropriate files to perform the analysis.

Produce a new Calcam calibration
--------------------------------

#. Save a reference frame from the chosen movie. Frames from large events such as disruptions often illuminate the
   most vessel structures, giving the best calibration images. Image enhancements such as histogram equalisation
   generally make features much easier to pick out.
   A `script in ir_tools <https://git.ccfe.ac.uk/mast-u-diagnostics/ir_tools/-/blob/dev/ir_tools/calcam_calibration
   /generate_calcam_calib_images.py#L373>`_ can be used to identify, enhance and
   save suitable calibration images for a given shot.
#. Perform the calcam calibration. If the image contains sufficiently many clearly identifiable points a point pair
   calibration is preferable, otherwise (as in the case of the MWIR radial views) a manual alignment calibration might
   be necessary. If a calibration already exists for this view, it can often be useful to load that calibration as a
   starting point, load your new calibration image and save the resulting calibration under a new filename.
   Calcam has good `online documentation <https://euratom-software.github.io/calcam/html/gui_intro.html>`_ to follow at
   this stage. Save the calibration with an informative name indicating the camera, shot number, calibration image and
   calibration quality (eg RMS pixel fit value).
#. Copy the new calcam calibration .ccc file to a location where FIRE can find it based on the calibration file paths
   in you fire_config.json. Typically this should be in the `air_calib` repository
   (e.g. air_calib/mast_u/calcam/calibrations) so that all calibrations are collected together. While the calibration
   image is saved in the .ccc calibration file, it is also a good idea to copy the calibration image that you used
   (e.g. to air_calib/mast_u/calcam/input_images) so that other users can see examples of calibration images or use them
   for their own calibrations.

Update the Calcam calibration lookup file
-----------------------------------------
The Calcam calibration lookup file now needs updating to tell FIRE which shots to use the new calcam calibration.
This is achieved by adding a new row to the appropriate file
(e.g. air_calib/mast_u/calcam_calibs-mast_u-rit-defaults.csv). Specify:

#. The stand *and* end (inclusive) shot range numbers for which the calibration should be used. Ensure the shot range
   doesn't overlap with that specified in any other rows as this will result in an error when the file is read. By
   specifying the end shot it is possible to leave shot ranges without a specified calibration file, so that trying to
   analyse a shot without a valid calibration will raise an error and prompt the user to produce a new calibration.
#. The *name* of the calibration file, including the .ccc extension, but excluding the file path.
#. The name of the the author of the calibration, so they can be contacted for clarification about it.
#. A comment describing the quality and context/reason for the updated calibration.

.. code-block::
    :caption: calcam_calibs-{machine}-{diag_tag_raw}-defaults.csv
    :emphasize-lines: 2
    :name: calcam-calibs-lookup
    pulse_start,pulse_end,calcam_calibration_file,author,comments
    43123,43647,rit_43141_218_nuc_eq_glob-v1.ccc,Tom Farley,Manual alignment calibration from first diverted plasmas

Define a new analysis path
--------------------------
The analysis path defines a set of ordered spatial points that are joined to produce an analysis path.
Many path definitions can be given in the same path definitions file (see e.g.
air_calib/mast_u/analysis_path_dfns-mast_u-rit-defaults.json).
Following json file formatting conventions each path definition is a top level 'object' (i.e. `"name": {...}`).
The 'name' is the string used to select the analysis path in the next step.
Each path definition is made up of:

#. A list of coordinate points.
#. A description string explaining the purpose/origin of the analysis path.

Each coordinate point on the analysis path should specify:

#. A name for the point e.g. "start"/"end".
#. The R, z and phi coordinates of the point.
#. The order of the point in the path sequence.
#. Whether to include the next interval between this point and the next in the sequence in the analysis path.
   Specifying `false` enables obstructions in the view to be skipped so the path jumps to another point in the image.

.. code-block:: json
    :caption: analysis_path_dfns-{machine}-{diag_tag_raw}-defaults.json
    :emphasize-lines: 2
    :name: calcam-calibs-lookup
    "MASTU_S3_lower_T2_radial_1":
    {
    "coords":
        [
            ["start", {"R": 0.771, "z": -1.743, "phi": 21.5, "order": 0, "include_next_interval": true}],
            ["end", {"R": 0.903, "z": -1.875, "phi": 21.5, "order": 1, "include_next_interval": false}]
        ],
    "description": "Outward radial path down lower divertor tile 2 in sector 3."
    },

Update the analysis path lookup file
------------------------------------
air_calib/mast_u/analysis_paths-mast_u-rit-defaults.csv

Update the camera settings file
-------------------------------
air_calib/mast_u/camera_settings-mast_u-rit.csv

Update the temperature (black body) calibration
-----------------------------------------------
air_calib/mast_u/temperature_coefs-mast_u-rit.csv

Update the material properties (THEORDOR) input file
----------------------------------------------------
air_calib/mast_u/material_props-mast_u-defaults.json

Produce an analysed UDA netcdf file
-----------------------------------

In the call to scheduler_workflow() you can specify `alpha_user` which will override the alpha parameter value
specified in the material properties file.

In the logging output a line will be printed that lists all the input settings files that are being used::
    INFO:fire.scheduler_workflow:scheduler_workflow:236:   Located input files for analysis: ...

Confirm that all the identified files are as expected.

Change alpha