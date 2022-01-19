from setuptools import setup, find_packages
import os, re, ast

_version_re = re.compile(r'__version__\s+=\s+(.*)')  # pattern of version number in root fire __init__.py file

# Get version number so it is only defined in one place
with open('fire/__init__.py', 'rb') as f:
    version = str(ast.literal_eval(_version_re.search(
        f.read().decode('utf-8')).group(1)))

setup(
    name='fire',
    version=version,
    description='Fusion Infra-Red Experiments analysis tool',
    url='git@git.ccfe.ac.uk/MAST-U_Scheduler/air',
    author='tfarley',
    author_email='tom.farley@ukaea.uk',
    # license=ccfepyutils.__license__,
    packages=['fire'],
    # packages=find_packages(exclude=['docs', 'external', 'misc', 'tests', 'third_party']),
    package_data={
        'fire': ['settings/*'],
    },
    include_package_data=True,
    install_requires=[
        "numpy>=1.20.1",
        "scipy>=1.1.0",
        "opencv-python >= 4.0.0.21",
        "pandas>=1.3.1",
        "xarray>=0.19.0",
        "scikit-image>=0.14.2",
        "netCDF4>=1.4.2",
        "matplotlib>=3.0.2",
        "natsort==5.5.0",
        "PyQt5==5.12",
        "future-fstrings<=1.2.0",
        "pyyaml",
        "pytest",
        "pytest-arraydiff>=0.3",
        "pytest-cov>=2.8.1",
        "pytest-doctestplus>=0.2.0",
        "Sphinx>=1.8.2",
        "sphinx-rtd-theme>=0.4.2",
        ],
    dependency_links=[
        "git+https://github.com/euratom-software/calcam.git",
        # "git+https://oauth2:$AUTH_TOKEN@git.ccfe.ac.uk/SOL_Transport/pyIpx.git@master"  # Requires ukaea github access
        ],
    # extras_require=['pyuda', 'mastvideo', 'pyIpx', 'pyEquilibrium', 'mastu_exhaust_analysis', 'ir_tools'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest-cov'],
    python_requires='>=3',
    test_suite='tests.test_suite_fast',  # 'tests.test_suite_slow'
    zip_safe=False,
long_description=open('README.md').read()
)