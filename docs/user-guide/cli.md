# Command-line interface

Zarr-Python provides a command-line interface that enables:

- migration of Zarr v2 metadata to v3
- removal of v2 or v3 metadata

To see available commands run the following in a terminal:

```bash
zarr --help
```

or to get help on individual commands:

```bash
zarr migrate --help

zarr remove-metadata --help
```

## Migrate metadata from v2 to v3

### Migrate to a separate location

To migrate a Zarr array/group's metadata from v2 to v3 run:

```bash
zarr migrate v3 path/to/input.zarr path/to/output.zarr
```

This will write new `zarr.json` files to `output.zarr`, leaving `input.zarr` un-touched.
Note - this will migrate the entire Zarr hierarchy, so if `input.zarr` contains multiple groups/arrays,
new `zarr.json` will be made for all of them.

### Migrate in-place

If you'd prefer to migrate the metadata in-place run:

```bash
zarr migrate v3 path/to/input.zarr
```

This will write new `zarr.json` files to `input.zarr`, leaving the existing v2 metadata un-touched.

To open the array/group using the new metadata use:

```python
import zarr
zarr_with_v3_metadata = zarr.open('path/to/input.zarr', zarr_format=3)
```

Once you are happy with the conversion, you can run the following to remove the old v2 metadata:

```bash
zarr remove-metadata v2 path/to/input.zarr
```

Note there is also a shortcut to migrate and remove v2 metadata in one step:

```bash
zarr migrate v3 path/to/input.zarr --remove-v2-metadata
```

## Remove metadata

Remove v2 metadata using:

```bash
zarr remove-metadata v2 path/to/input.zarr
```

or v3 with:

```bash
zarr remove-metadata v3 path/to/input.zarr
```

By default, this will only allow removal of metadata if a valid alternative exists. For example, you can't
remove v2 metadata unless v3 metadata exists at that location.

To override this behaviour use `--force`:

```bash
zarr remove-metadata v3 path/to/input.zarr --force
```

## Dry run

All commands provide a `--dry-run` option that will log changes that would be made on a real run, without creating
or modifying any files.

```bash
zarr migrate v3 path/to/input.zarr --dry-run

Dry run enabled - no new files will be created or changed. Log of files that would be created on a real run:
Saving metadata to path/to/input.zarr/zarr.json
```

## Verbose

You can also add `--verbose` **before** any command, to see a full log of its actions:

```bash
zarr --verbose migrate v3 path/to/input.zarr

zarr --verbose remove-metadata v2 path/to/input.zarr
```

## Equivalent functions

All features of the command-line interface are also available via functions under
`zarr.metadata`.