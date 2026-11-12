`ZarrV2ArrayMetadata.create_default(shape=...)` derives a chunk length of 1 for a dimension of length 0, as the v3 model does, where it wrote 0, which zarr-python repairs to 1 on read.
