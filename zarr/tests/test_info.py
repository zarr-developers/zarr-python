# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division


import zarr
import numcodecs


def test_info():

    # setup
    g = zarr.group(store=dict(), chunk_store=dict(),
                   synchronizer=zarr.ThreadSynchronizer())
    g.create_group('foo')
    z = g.zeros('bar', shape=10, filters=[numcodecs.Adler32()])

    # test group info
    items = g.info_items()
    keys = sorted([k for k, _ in items])
    expected_keys = sorted([
        'Type', 'Read-only', 'Synchronizer type', 'Store type', 'Chunk store type',
        'No. members', 'No. arrays', 'No. groups', 'Arrays', 'Groups', 'Name'
    ])
    assert expected_keys == keys

    # test array info
    items = z.info_items()
    keys = sorted([k for k, _ in items])
    expected_keys = sorted([
        'Type', 'Data type', 'Shape', 'Chunk shape', 'Order', 'Read-only', 'Filter [0]',
        'Compressor', 'Synchronizer type', 'Store type', 'Chunk store type', 'No. bytes',
        'No. bytes stored', 'Storage ratio', 'Chunks initialized', 'Name'
    ])
    assert expected_keys == keys
