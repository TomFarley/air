import unittest
from pathlib import Path

try:
	from fire.io.uda_io import return_true

	ipx_path = Path('test_data/mast/')

	# @pytest.fixture  # Run function once and save output to supply to multiple tests
	# def expected_ouput():
	# 	aa=25
	# 	bb =35
	# 	cc=45
	# 	return [aa,bb,cc]

	class TestUdaIO(unittest.TestCase):

		def test_return_true(self):
			self.assertTrue(return_true())

except ImportError as e:
	# Do not define test class if cannot import pyuda
	# TODO: Find correct way of selectively running tests depending on available modules!
	pass


