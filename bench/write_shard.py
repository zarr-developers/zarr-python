import itertools
import os.path
import shutil
import sys
import tempfile
import timeit

import line_profiler
import numpy as np

import zarr
import zarr.codecs
import zarr.codecs.sharding

if __name__ == "__main__":
    sys.path.insert(0, "..")

    # setup
    with tempfile.TemporaryDirectory() as path:

        ndim = 3
        opt = {
            'shape': [1024]*ndim,
            'chunks': [128]*ndim,
            'shards': [512]*ndim,
            'dtype': np.float64,
        }

        store = zarr.storage.LocalStore(path)
        z = zarr.create_array(store, **opt)
        print(z)

        def cleanup() -> None:
            for elem in os.listdir(path):
                elem = os.path.join(path, elem)
                if not elem.endswith(".json"):
                    if os.path.isdir(elem):
                        shutil.rmtree(elem)
                    else:
                        os.remove(elem)

        def write() -> None:
            wchunk = [512]*ndim
            nwchunks = [n//s for n, s in zip(opt['shape'], wchunk, strict=True)]
            for shard in itertools.product(*(range(n) for n in nwchunks)):
                slicer = tuple(
                    slice(i*n, (i+1)*n)
                    for i, n in zip(shard, wchunk, strict=True)
                )
                d = np.random.rand(*wchunk).astype(opt['dtype'])
                z[slicer] = d

        print("*" * 79)

        # time
        vars = {"write": write, "cleanup": cleanup, "z": z, "opt": opt}
        t = timeit.repeat("write()", "cleanup()", repeat=2, number=1, globals=vars)
        print(t)
        print(min(t))
        print(z)

        # profile
        # f = zarr.codecs.sharding.ShardingCodec._encode_partial_single
        # profile = line_profiler.LineProfiler(f)
        # profile.run("write()")
        # profile.print_stats()
