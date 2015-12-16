# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division
from glob import glob
import os
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


import numpy as np
include_dirs.append(np.get_include())


from Cython.Build import cythonize
ext_modules = cythonize([
    Extension('zarr.ext',
              sources=['zarr/ext.pyx'] + blosc_sources,
              include_dirs=include_dirs,
              define_macros=define_macros,
              ),
])


setup(
    name='zarr',
    description='TODO',
    long_description='TODO',
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
)
