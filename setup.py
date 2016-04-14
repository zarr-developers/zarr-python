# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from glob import glob
import os
from setuptools import setup, Extension
import cpuinfo
import numpy as np
from Cython.Build import cythonize


# setup
blosc_sources = []
extra_compile_args = []
include_dirs = []
define_macros = []

# generic setup
blosc_sources += [f for f in glob('c-blosc/blosc/*.c')
                  if 'avx2' not in f and 'sse2' not in f]
blosc_sources += glob('c-blosc/internal-complibs/lz4*/*.c')
blosc_sources += glob('c-blosc/internal-complibs/snappy*/*.cc')
blosc_sources += glob('c-blosc/internal-complibs/zlib*/*.c')
include_dirs += [os.path.join('c-blosc', 'blosc')]
include_dirs += glob('c-blosc/internal-complibs/*')
define_macros += [('HAVE_LZ4', 1), ('HAVE_SNAPPY', 1), ('HAVE_ZLIB', 1)]
include_dirs.append(np.get_include())

# determine CPU support for SSE2 and AVX2
cpu_info = cpuinfo.get_cpu_info()

# SSE2
if 'sse2' in cpu_info['flags']:
    print('SSE2 detected')
    extra_compile_args.append('-DSHUFFLE_SSE2_ENABLED')
    extra_compile_args.append('-msse2')
    blosc_sources += [f for f in glob('c-blosc/blosc/*.c') if 'sse2' in f]

# AVX2
if 'avx2' in cpu_info['flags']:
    print('AVX2 detected')
    extra_compile_args.append('-DSHUFFLE_AVX2_ENABLED')
    extra_compile_args.append('-mavx2')
    blosc_sources += [f for f in glob('c-blosc/blosc/*.c') if 'avx2' in f]

# define extension module
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
    install_requires=[
        'fasteners',
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
