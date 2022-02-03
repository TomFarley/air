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
        'alabaster==0.7.12',
        'attrs==21.4.0',
        'Babel==2.9.1',
        'backcall==0.2.0',
        'certifi==2021.10.8',
        'cftime==1.5.1.1',
        'charset-normalizer==2.0.10',
        'cloudpickle==2.0.0',
        'coverage==6.2',
        'cycler==0.11.0',
        'dask==2022.1.0',
        'decorator==5.1.1',
        'docutils==0.17.1',
        'fonttools==4.28.5',
        'fsspec==2022.1.0',
        'future-fstrings==1.2.0',
        'Glymur==0.9.7.post1',
        'idna==3.3',
        'imageio==2.13.5',
        'imagesize==1.3.0',
        'importlib-metadata==4.10.1',
        'iniconfig==1.1.1',
        'ipython==7.31.0',
        'jedi==0.18.1',
        'Jinja2==3.0.3',
        'kiwisolver==1.3.2',
        'locket==0.2.1',
        'lxml==4.7.1',
        'MarkupSafe==2.0.1',
        'matplotlib==3.5.1',
        'matplotlib-inline==0.1.3',
        'natsort==5.5.0',
        'netCDF4==1.4.2',
        'networkx==2.6.3',
        'numpy==1.21.5',
        'opencv-python==4.0.0.21',
        'packaging==21.3',
        'pandas==1.3.5',
        'parso==0.8.3',
        'partd==1.2.0',
        'pexpect==4.8.0',
        'pickleshare==0.7.5',
        'Pillow==9.0.0',
        'pluggy==1.0.0',
        'prompt-toolkit==3.0.24',
        'ptyprocess==0.7.0',
        'py==1.11.0',
        'Pygments==2.11.2',
        'pyparsing==3.0.6',
        'PyQt5==5.12',
        'PyQt5_sip==4.19.19',
        'pytest==6.2.5',
        'pytest-arraydiff==0.3',
        'pytest-cov==2.8.1',
        'pytest-doctestplus==0.2.0',
        'python-dateutil==2.8.2',
        'pytz==2021.3',
        'PyWavelets==1.2.0',
        'PyYAML==6.0',
        'requests==2.27.1',
        'scikit-image==0.18.3',
        'scipy==1.1.0',
        'six==1.16.0',
        'snowballstemmer==2.2.0',
        'Sphinx==4.4.0',
        'sphinx-rtd-theme==0.4.2',
        'sphinxcontrib-applehelp==1.0.2',
        'sphinxcontrib-devhelp==1.0.2',
        'sphinxcontrib-htmlhelp==2.0.0',
        'sphinxcontrib-jsmath==1.0.1',
        'sphinxcontrib-qthelp==1.0.3',
        'sphinxcontrib-serializinghtml==1.1.5',
        'sphinxcontrib-websupport==1.2.4',
        'tifffile==2021.11.2',
        'toml==0.10.2',
        'toolz==0.11.2',
        'traitlets==5.1.1',
        'typing_extensions==4.0.1',
        'urllib3==1.26.8',
        'wcwidth==0.2.5',
        'xarray==0.19.0',
        'zipp==3.7.0',
        ],
    dependency_links=[
        "git+https://github.com/euratom-software/calcam.git",
        # "Calcam @ git+https://github.com/euratom-software/calcam.git@7db9b5aed127fde5974715586be723d7a5c95809"
        # -e git+https://tfarley@git.ccfe.ac.uk/SOL_Transport/pyIpx.git@a841328bffd952620c4e0ce56a7248e81181d9b0#egg=pyIpx
        # -e git+ssh://git@git.ccfe.ac.uk/SOL_Transport/pyEquilibrium.git@daecc3b5f1059d2523ea18f63e4fd0368b8cb030#egg=pyEquilibrium
        # -e git+ssh://git@git.ccfe.ac.uk/jrh/mastu_exhaust_analysis.git@bf9c682ccd5c22cedb4ca03995b5fb685e78bc4c#egg=mastu_exhaust_analysis
        # -e git+ssh://git@git.ccfe.ac.uk/MAST-U/mastvideo.git@e14465866cd26e268b090844e8efa921d06eeaa4#egg=mastvideo
        # -e git+ssh://git@git.ccfe.ac.uk/tfarley/ir_tools.git@1e8ebf0a017a5166635555570bb356b7bddd0236#egg=ir_tools

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