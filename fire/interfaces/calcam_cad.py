#!/usr/bin/env python

"""


Created: 
"""

import logging
from typing import Union, Iterable, Sequence, Tuple, Optional, Any, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

import vtk

logger = logging.getLogger(__name__)
logger.propagate = False


def get_cell_locator(cad_model):
    '''
    Get a vtkCellArray object used for getting tile surfaces.

    Returns:

        vtk.vtkCellArray : VTK cell array.
    '''
    vtk_type = vtk.vtkCellArray
    # vtkLookupTable, vtkPoints, vtkPolyDataNormals, vtkPolyDataTangents, vtkPolyVertex

    # Don't return anything if we have no enabled geometry
    if len(cad_model.get_enabled_features()) == 0:
        return None

    if cad_model.cell_locator is None:

        appender = vtk.vtkAppendPolyData()

        for fname in cad_model.get_enabled_features():
            appender.AddInputData(cad_model.features[fname].get_polydata())

        appender.Update()

        cad_model.cell_locator = vtk_type()
        cad_model.cell_locator.SetTolerance(1e-6)
        cad_model.cell_locator.SetDataSet(appender.GetOutput())
        cad_model.cell_locator.BuildLocator()

    return cad_model.cell_locator

if __name__ == '__main__':
    pass