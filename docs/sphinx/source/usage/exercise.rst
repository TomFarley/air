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
A `script in ir_tools<| * Output directory for saving checkpoint cache files>`_ can be used to identify, enhance and
save suitable calibration images for a given shot.

#. Perform the calcam calibration. If the image contains sufficiently many clearly identifiable points a point pair
calibration is preferable, otherwise (as in the case of the MWIR radial views) a manual alignment calibration might
be necessary. If a calibration already exists for this view, it can often be useful to load that calibration as a
starting point, load your new calibration image and save the resulting calibration under a new filename.
Calcam has good `online documentation<https://euratom-software.github.io/calcam/html/gui_intro.html>`_ to follow at
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
(e.g. air_calib/mast_u/calcam_calibs-mast_u-rit-defaults.csv). Speccify:
#. The stand *and* end (inclusive) shot range numbers for which the calibration should be used. Ensure the shot range
 doesn't overlap with that specified in any other rows as this will result in an error when the file is read. By
 specifying the end shot it is possible to leave shot ranges without a specified calibration file, so that trying to
 analyse a shot without a valid calibration will raise an error and prompt the user to produce a new calibration.
#. The *name* of the calibration file, including the .ccc extension, but excluding the file path.
#. The name of the the author of the calibration, so they can be contacted for clarification about it.
#. A comment describing the quality and context/reason for the updated calibration.

Define a new analysis path
--------------------------

Update the analysis path lookup file
------------------------------------

Update the camera settings file
-------------------------------

Update the material properties (THEORDOR) input file
----------------------------------------------------

Produce an analysed UDA netcdf file
-----------------------------------

Add analysis path
Camera settings
Change alpha