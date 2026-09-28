`ZarrV3ArrayMetadata.create_default(shape=...)` derives a chunk length of 1
for a dimension of length 0, where it wrote 0: the regular grid asks for
chunk lengths greater than zero, and `zarr` does not open a grid with one.
