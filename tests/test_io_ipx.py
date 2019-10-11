import pytest
import unittest
from pathlib import Path

from fire.io.ipx_io import get_ipx_meta_data, get_ipx_frames, return_true

ipx_path = Path('test_data/mast/')

# @pytest.fixture  # Run function once and save output to supply to multiple tests
# def expected_ouput():
# 	aa=25
# 	bb =35
# 	cc=45
# 	return [aa,bb,cc]

class TestIpxIO(unittest.TestCase):

	# def test_get_ipx_meta_data(self):
	# 	ipx_fn = 'rir030378.ipx'
	# 	ipx_path_fn = ipx_path / ipx_fn
	# 	ipx_meta_data = get_ipx_meta_data(ipx_path_fn)
	# 	self.assertTrue(isinstance(ipx_meta_data, dict))

	def test_return_true(self):
		self.assertTrue(return_true())



