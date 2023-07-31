#!/usr/bin/env python
try:
    from setuptools import setup, Extension
except ImportError:
    raise RuntimeError('setuptools is required')

import os

from distutils.util import convert_path
import sys

# The minimum python version which can be used to run ObsPy
MIN_PYTHON_VERSION = (3, 7)

# Fail fast if the user is on an unsupported version of python.
if sys.version_info < MIN_PYTHON_VERSION:
    msg = ("cleanbf requires python version >= {}".format(MIN_PYTHON_VERSION) +
           " you are using python version {}".format(sys.version_info))
    print(msg, file=sys.stderr)
    sys.exit(1)


DESCRIPTION = 'A set of functions for running CLEAN beamforming.'
LONG_DESCRIPTION = """
"""

DISTNAME = 'cleanbf'
LICENSE = 'GPL-3.0'
AUTHOR = 'Jake Anderson'
MAINTAINER_EMAIL = 'ajakef@gmail.com'
URL = 'https://github.com/ajakef/clean'


VERSION = '0.2'

## Dependency notes:
# pandas: >= 1.3 
# numpy: >=1.22 (earlier versions have security issue https://github.com/advisories/GHSA-fpfv-jqm9-f5jm)
# python: >=3.8 (>=3.8 is required by numpy 1.22).
# obspy: >=1.3 (>=1.3 is required for numpy 1.22 compatibility)

## example range: pandas>=1.3.0,<1.4.0
## example exact: numpy==1.21
INSTALL_REQUIRES = [
    'obspy>=1.2.2', 
    'numpy>=1.22', 
    'pandas>=1.3.0', 
    'scipy>=1.3.0', 
    'matplotlib>=3.2.0', 
]

TESTS_REQUIRE = ['pytest']

EXTRAS_REQUIRE = {
    # 'optional': [...],
    'doc': ['sphinx==3.2.1', 'sphinx-rtd-theme==0.5.0'],
    'test': TESTS_REQUIRE
}
EXTRAS_REQUIRE['all'] = sorted(set(sum(EXTRAS_REQUIRE.values(), [])))

CLASSIFIERS = [
    #'Development Status :: 4 - Beta',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Physics'
]

ENTRY_POINTS = {
    'console_scripts': [
    ]
}

setuptools_kwargs = {
    'zip_safe': False,
    'scripts': [],
    'include_package_data': True,
    #'package_data':{'cleanbf':['cleanbf/data/*', 'demos/*']}
}

PACKAGES = ['cleanbf']

setup(name=DISTNAME,
      setup_requires=['setuptools>=18.0'],
      version=VERSION,
      packages=PACKAGES,
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE,
      tests_require=TESTS_REQUIRE,
      ext_modules=[],
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      maintainer_email=MAINTAINER_EMAIL,
      license=LICENSE,
      url=URL,
      classifiers=CLASSIFIERS,
      entry_points=ENTRY_POINTS,
      **setuptools_kwargs)
