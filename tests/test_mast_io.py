import pytest
from pathlib import Path

from fire.io.mast_io import read_local_ipx_file

ipx_path = Path('test_data/mast/')

@pytest.fixture  # Run function once and save output to supply to multiple tests
def expected_ouput():
	aa=25
	bb =35
	cc=45
	return [aa,bb,cc]

@pytest.mark.set1
def test_read_local_ipx_file(expected_ouput):
	ipx_fn = 'rir030378.ipx'
	ipx_path_fn = ipx_path / ipx_fn