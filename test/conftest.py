"""
Common test configuration for all test subdirectories.
Put here only those things that can not be done through command line options and pytest.ini file.
"""


import sys
import os
import pytest

from test.fixtures.mlmc_test_run import TestMLMC

# Modify sys.path to have path to the source dir. This allow to run tests from sources
# without virtual environment and without installation of the package.
# Try to remove this as we
# TODO: make installation and Tox working in order to remove this hack.
this_source_dir = os.path.dirname(os.path.realpath(__file__))
rel_paths = ["../src"]
for rel_path in rel_paths:
    sys.path.append(os.path.realpath(os.path.join(this_source_dir, rel_path)))
sys.path = [ x for x in sys.path if x not in {this_source_dir, ''} ]
print(sys.path)

#https://stackoverflow.com/questions/37563396/deleting-py-test-tmpdir-directory-after-successful-test-case
# @pytest.fixture(scope='session')
# def temporary_dir(tmpdir_factory):
#     img = compute_expensive_image()
#     fn = tmpdir_factory.mktemp('data').join('img.png')
#     img.save(str(fn))
#     return fn