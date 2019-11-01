"""Combining Test Suites - unittest""

#  ----------------- Unit test framework
import unittest

#  ----------------  Individual test suites
import Traintest
import Stationtest

# -----------------  Load all the test cases
suiteList = []
suiteList.append(unittest.TestLoader().loadTestsFromTestCase(Traintest.TrainTest))
suiteList.append(unittest.TestLoader().loadTestsFromTestCase(Stationtest.StationTest))

# ----------------   Join them together ane run them
comboSuite = unittest.TestSuite(suiteList)
unittest.TextTestRunner(verbosity=0).run(comboSuite)