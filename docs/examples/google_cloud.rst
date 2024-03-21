Opening from Google Cloud Storage
=================================
This example shows how to open a read-only bucket in Google Cloud Storage.
This requires the ``gcsfs`` package in addition to ``zarr``.


.. code:: python

    import zarr
    import gcsfs

    project = "my_project"
    bucket = "my_bucket"
    path_to_array = "a/b/c"

    fs = gcsfs.GCSFileSystem(project=bucket, token='anon', access='read_only')
    store = zarr.FSStore(url=bucket, fs=fs)
    array = zarr.open_array(store=store, path=path_to_array)
