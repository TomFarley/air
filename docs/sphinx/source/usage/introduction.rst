====================
Introduction to FIRE
====================

Purpose of FIRE
---------------
FIRE is designed to be a flexible tool for analysing infra-red camera images of divertor target tiles in tokamaks,
using the THEODOR code to calculate heat fluxes to the tile surfaces.
The code has initially been written with application in the MAST-Upgrade and JET tokamaks, but it's modularity and
plugin system should make it's use on other machines straighforward.
This should make comparisons between different machines easier to interpret, when each machine has been analysed in a
consistent way.

Input files required to run FIRE
--------------------------------
Machine and camera specific variables are stored in a number of input files in human readable json, or csv files.

.. list-table:: Input files
   :widths: 15 5 30 50
   :header-rows: 1

   * - File type
     - Description
     - Extension
     - Default filename format
   * - machine_plugins
     - | Functions sepecific to machine (e.g. s-coords, sectors
       | etc.). File must be in a plugin search path and contain
       | the attribute 'machine_plugin_name'.
     - py
     - {machine}.py  (arb)
   * - calcam_calibs
     - | Lookup table for which calcam spatial calibration files
       | (.ccc) to use for different pulse ranges
     - csv
     - calcam_calibs-{machine}-{diag_tag_raw}-defaults.csv
   * - analysis_paths
     - | Spatial coordinates of paths along tile surfaces to
       | calculate heat fluxes along
     - csv
     - analysis_paths-{machine}-{diag_tag_raw}-defaults.csv
