import pytest
import unittest
from pathlib import Path

from fire.io.mast_io import get_ipx_meta_data, get_ipx_frames, get_frames_uda, return_true

ipx_path = Path('test_data/mast/')

# @pytest.fixture  # Run function once and save output to supply to multiple tests
# def expected_ouput():
# 	aa=25
# 	bb =35
# 	cc=45
# 	return [aa,bb,cc]

class TestMastIO(unittest.TestCase):

# @pytest.mark.set1
# def test_get_ipx_meta_data(expected_ouput):
# 	ipx_fn = 'rir030378.ipx'
# 	ipx_path_fn = ipx_path / ipx_fn
# 	ipx_meta_data = get_ipx_meta_data(ipx_path_fn)
# 	assert isinstance(ipx_meta_data, dict)

	def test_return_true(self):
		self.assertTrue(return_true())


