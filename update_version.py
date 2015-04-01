""" file: update_version.py
    author: Jess Robertson
            CSIRO Minerals Resources Flagship
    date:   January 2015

    description: Tool to update the version number using `git describe`
"""

import subprocess
import os
from setuptools import Command
import re
import logging

LOGGER = logging.getLogger('pysiss')

VERSION_PY_TEMPLATE = """\
# This file is originally generated from Git information by running 'setup.py
# update_version'. Distribution tarballs contain a pre-generated copy of this
# file.
__version__ = '{0}'
"""


def update_version():
    # Query git for the current description
    if not os.path.isdir(".git"):
        LOGGER.warn("This does not appear to be a Git repository, leaving "
                    "pysiss/_version.py alone.")
        return
    try:
        p = subprocess.Popen(["git", "describe", "--always"],
                             stdout=subprocess.PIPE)
        stdout = p.communicate()[0]
        if p.returncode != 0:
            raise EnvironmentError
        else:
            ver = stdout.strip().split('-')
            if len(ver) > 1:
                ver = ver[0] + '.dev' + ver[1]
            else:
                ver = ver[0]
    except EnvironmentError:
        LOGGER.warn(
            "Unable to run git, leaving pysiss/_version.py alone")
        return

    # Write to file
    current_ver = get_version()
    if current_ver != ver:
        LOGGER.info("Version {0} out of date, updating to {1}".format(
            current_ver, ver))
        with open('pysiss/_version.py', 'wb') as fhandle:
            fhandle.write(VERSION_PY_TEMPLATE.format(ver))


def get_version():
    """ Get the currently set version
    """
    try:
        with open("pysiss/_version.py") as fhandle:
            for line in (f for f in fhandle if not f.startswith('#')):
                return re.match("__version__ = '([^']+)'", line).group(1)
    except EnvironmentError:
        LOGGER.error(
            "Can't find pysiss/_version.py - what's the version, doc?")
        return 'unknown'


class Version(Command):

    """ Setuptools command to update version

        to execute run python setup.py version
    """

    description = "update _version.py from Git repo"
    user_options = []
    boolean_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        update_version()
        LOGGER.info("Version is now {0}".format(get_version()))

if __name__ == '__main__':
    update_version()
