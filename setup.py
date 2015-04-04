#!/usr/bin/env python
""" file: setup.py (pysiss)
    author: Jess Robertson, CSIRO Earth Science and Resource Engineering
    date: Wednesday 1 April, 2015

    description: Setuptools installer script for pymaxion.
"""

from setuptools import setup, find_packages
import os

def read(*paths):
    """ Build a file path from *paths and return the contents.
    """
    with open(os.path.join(*paths), 'r') as f:
        return f.read()

# Get requirements from requirements.txt file
with open('requirements.txt') as fhandle:
    REQUIREMENTS = [l.strip('\n') for l in fhandle]

# Get version number from _version.py
# Can be updated using python setup.py update_version
from update_version import update_version, Version, get_version
update_version()

## PACKAGE INFORMATION
setup(
    # Metadata
    name='pymaxion',
    version=get_version(),
    description='Polygon projections in Python for Shapely',
    long_description=read('README.md'),
    author='Jess Robertson',
    author_email='jesse.robertson@csiro.au',
    url='http://github.com/jesserobertson/pymaxion',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2 :: Only',
        'Topic :: Internet',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: GIS',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],

    # Dependencies
    install_requires=REQUIREMENTS,

    # Contents
    packages=find_packages(exclude=['test*']),
    package_data={
        'pymaxion.resources': ['*']
    },
    test_suite='tests',
    cmdclass={
        'update_version': Version,
    }
)
