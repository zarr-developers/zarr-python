import sys
import timeit

import numpy as np

import line_profiler
import zarr
from zarr import blosc

if __name__ == "__main__":

    sys.path.insert(0, '..')

    # setup
    a = np.random.normal(2000, 1000, size=200000000).astype('u2')
    z = zarr.empty_like(a, chunks=1000000,
                        compression='blosc',
                        compression_opts=dict(cname='lz4', clevel=5, shuffle=2))
    print(z)

    print('*' * 79)

    # time
    t = timeit.repeat('z[:] = a', repeat=10, number=1, globals=globals())
    print(t)
    print(min(t))
    print(z)

    # profile
    profile = line_profiler.LineProfiler(blosc.compress)
    profile.run('z[:] = a')
    profile.print_stats()

    print('*' * 79)

    # time
    t = timeit.repeat('z[:]', repeat=10, number=1, globals=globals())
    print(t)
    print(min(t))

    # profile
    profile = line_profiler.LineProfiler(blosc.decompress)
    profile.run('z[:]')
    profile.print_stats()
