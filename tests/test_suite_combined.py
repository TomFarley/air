"""Combining Test Suites - unittest

For inmports to resolve, run test suite from top level air directory (not from tests directory) ie:
~/repos/air $ ipython tests/test_suite_fast.py    OR
~/repos/air $ python setup.py test
"""

#  ----------------- Unit test framework
import unittest

#  ----------------  Individual test suites
from tests import test_suite_slow
from tests import test_suite_fast

# -----------------  Load all the test cases
suiteList = []
suiteList.append(unittest.TestLoader().loadTestsFromModule(test_suite_slow))
# suiteList.append(unittest.TestLoader().loadTestsFromTestCase(Stationtest.StationTest))

# ----------------   Join them together ane run them
comboSuite = unittest.TestSuite(suiteList)
unittest.TextTestRunner(verbosity=0).run(comboSuite)