# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from glob import glob
import os
import re
import platform
import ctypes
from setuptools import setup, Extension


blosc_sources = []
include_dirs = []
define_macros = []


# We still have to figure out how to detect AVX2 in Python,
# so no AVX2 support for the time being
blosc_sources += [f for f in glob('c-blosc/blosc/*.c') if 'avx2' not in f]
blosc_sources += glob('c-blosc/internal-complibs/lz4*/*.c')
blosc_sources += glob('c-blosc/internal-complibs/snappy*/*.cc')
blosc_sources += glob('c-blosc/internal-complibs/zlib*/*.c')
include_dirs += [os.path.join('c-blosc', 'blosc')]
include_dirs += glob('c-blosc/internal-complibs/*')
define_macros += [('HAVE_LZ4', 1), ('HAVE_SNAPPY', 1), ('HAVE_ZLIB', 1)]
define_macros += [('CYTHON_TRACE', 1)]


extra_compile_args = []
if re.match('i.86|x86|AMD', platform.machine()) is not None:
    # always enable SSE2 for AMD/Intel machines
    extra_compile_args.append('-DSHUFFLE_SSE2_ENABLED')

is_32bit = ctypes.sizeof(ctypes.c_voidp) == 4
if is_32bit:
    if os.name == 'posix':
        extra_compile_args.append('-msse2')
    elif os.name == 'nt':
        # this is currently broken for windows
        extra_compile_args.append('/arch:sse2')


import numpy as np
include_dirs.append(np.get_include())


from Cython.Build import cythonize
ext_modules = cythonize([
    Extension('zarr.ext',
              sources=['zarr/ext.pyx'] + blosc_sources,
              include_dirs=include_dirs,
              define_macros=define_macros,
              extra_compile_args=extra_compile_args,
              ),
])

description = 'A minimal implementation of chunked, compressed, ' \
              'N-dimensional arrays for Python.'

with open('README.rst') as f:
    long_description = f.read()

setup(
    name='zarr',
    description=description,
    long_description=long_description,
    use_scm_version={
        'version_scheme': 'guess-next-dev',
        'local_scheme': 'dirty-tag',
        'write_to': 'zarr/version.py'
    },
    setup_requires=[
        'cython>=0.22',
        'numpy>=1.7',
        'setuptools>18.0',
        'setuptools-scm>1.5.4'
    ],
    ext_modules=ext_modules,
    package_dir={'': '.'},
    packages=['zarr', 'zarr.tests'],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    author='Alistair Miles',
    author_email='alimanfoo@googlemail.com',
    maintainer='Alistair Miles',
    maintainer_email='alimanfoo@googlemail.com',
    url='https://github.com/alimanfoo/zarr',
    license='MIT',
)
