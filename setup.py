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
        "numpy >= 1.12.0",
        "scipy",
        "xarray",
        "pandas",
        "matplotlib",
        "opencv-python",
        "future-fstrings",  # tmp
        ],
    python_requires='>=3',
    setup_requires=['pytest-runner'],
    test_suite='tests.test_suite_fast',  # 'tests.test_suite_slow'
    tests_require=['pytest-cov'],
    zip_safe=False,
long_description=open('README.md').read()
)