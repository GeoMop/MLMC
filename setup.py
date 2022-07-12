#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import io
import re
import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup

__version__ = '1.0.2'


# For long description:
def read(*names, **kwargs):
    return io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ).read()

long_description='%s\n%s' % (
    re.compile('^.. start-badges.*^.. end-badges', re.M | re.S).sub('', read('README.rst')),
    re.sub(':[a-z]+:`~?(.*?)`', r'``\1``', read('CHANGES.rst'))
)

setup(
    # Project (package) name
    name='mlmc',
    version=__version__,
    license='GPL 3.0',
    description='Multilevel Monte Carlo method.',
    long_description=long_description,
    author='Jan Brezina, Martin Spetlik',
    author_email='jan.brezina@tul.cz',
    url='https://github.com/GeoMop/MLMC',
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        # uncomment if you test on these interpreters:
        # 'Programming Language :: Python :: Implementation :: IronPython',
        # 'Programming Language :: Python :: Implementation :: Jython',
        # 'Programming Language :: Python :: Implementation :: Stackless',
        'Topic :: Scientific/Engineering',
    ],
    keywords=[
        # eg: 'keyword1', 'keyword2', 'keyword3',
    ],
    # packages=find_packages(where='.',
    #     exclude=['examples*', 'test*', 'docs']),
    packages=['mlmc', 'mlmc.plot', 'mlmc.quantity', 'mlmc.random', 'mlmc.sim', 'mlmc.tool'],
    # include automatically all files in the template MANIFEST.in
    include_package_data=True,
    zip_safe=False,
    install_requires=['numpy', 'scipy', 'sklearn', 'h5py>=3.1.0', 'ruamel.yaml', 'attrs', 'gstools', 'memoization'],
)
