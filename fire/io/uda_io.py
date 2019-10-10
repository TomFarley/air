#!/usr/bin/env python

""" 
Author: T. Farley
"""

import logging, os, itertools, re, inspect, configparser, time
from collections import defaultdict, OrderedDict
from datetime import datetime
from copy import copy, deepcopy
from pathlib import Path
from logging.config import fileConfig

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from nested_dict import nested_dict

from ccfepytools.utils import make_itterable, remove_duplicates_from_list, is_subset, get_methods_class
from ccfepytools.classes.plot import Plot

# fileConfig('../logging_config.ini')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Name(object):
    """ """

    def __init__(self):
        pass

    def __repr__(self):
        class_name = re.search(".*\.(\w+)'\>", str(self.__class__)).groups()[0]
        return '<{}: {}>'.format(class_name, None)


if __name__ == '__main__':
    pass