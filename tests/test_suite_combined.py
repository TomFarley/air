"""Combining Test Suites - unittest"""

#  ----------------- Unit test framework
import unittest

#  ----------------  Individual test suites
from . import test_suite_slow
from . import test_suite_fast

# -----------------  Load all the test cases
suiteList = []
suiteList.append(unittest.TestLoader().loadTestsFromTestCase(test_suite_slow))
# suiteList.append(unittest.TestLoader().loadTestsFromTestCase(Stationtest.StationTest))

# ----------------   Join them together ane run them
comboSuite = unittest.TestSuite(suiteList)
unittest.TextTestRunner(verbosity=0).run(comboSuite)