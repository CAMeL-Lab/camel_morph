# -*- coding: utf-8 -*-

# MIT License
#
# Copyright 2018-2022 New York University Abu Dhabi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
from setuptools import setup
import sys


VERSION_FILE = os.path.join(os.path.dirname(__file__),
                            'camel_morph',
                            'VERSION')
with open(VERSION_FILE, encoding='utf-8') as version_fp:
    VERSION = version_fp.read().strip()


CLASSIFIERS = [
    'Development Status :: 5 - Production/Stable',
    'Environment :: Console',
    'Intended Audience :: Developers',
    'Intended Audience :: Education',
    'Intended Audience :: Information Technology',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: MIT License',
    'Natural Language :: Arabic',
    'Topic :: Text Processing',
    'Topic :: Text Processing :: Linguistic',
]

DESCRIPTION = ('An environment for creating, and debugging morphological grammars '
               'that can be used in conjunction with morphological analyzers/generators'
               'developed by the CAMeL Lab at New York University Abu Dhabi.')

README_FILE = os.path.join(os.path.dirname(__file__), 'README.md')
with open(README_FILE, 'r', encoding='utf-8') as version_fp:
    LONG_DESCRIPTION = version_fp.read().strip()

INSTALL_REQUIRES = [
    'flask',
    'gspread==5.1.1',
    'numpy',
    'pandas',
    'tqdm',
    'pyrsistent',
    'emoji'
]

setup(
    name='camel_morph',
    version=VERSION,
    author='Christian Khairallah',
    author_email='christian.khairallah@nyu.edu',
    maintainer='Christian Khairallah',
    maintainer_email='christian.khairallah@nyu.edu',
    packages=['camel_morph',
              'camel_morph.debugging',
              'camel_morph.eval',
              'camel_morph.sandbox',
              'camel_morph.utils',
              'camel_morph.camel_tools',
              'camel_morph.camel_tools.camel_tools.cli',
              'camel_morph.camel_tools.camel_tools.utils',
              'camel_morph.camel_tools.camel_tools.morphology',
              'camel_morph.camel_tools.camel_tools.disambig',
              'camel_morph.camel_tools.camel_tools.tokenizers',
              'camel_morph.camel_tools.camel_tools.data'],
    package_data={
        'camel_morph': ['configs/*.json'],
    },
    include_package_data=True,
    url='https://github.com/CAMeL-Lab/camel_morph',
    license='MIT',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    classifiers=CLASSIFIERS,
    install_requires=INSTALL_REQUIRES,
    python_requires='>=3.8, <3.10'
)