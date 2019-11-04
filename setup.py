from setuptools import setup, find_packages
import os

setup(name='fire',
      version='2.0',
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
      install_requires=["numpy >= 1.12.0", "scipy", "xarray", "pandas"],
      python_requires='>=3',
      setup_requires=['pytest-runner'],
      test_suite='tests.test_suite',
      tests_require=['pytest-cov'],
      zip_safe=False,
      long_description=open('README.md').read())