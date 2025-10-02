# Zarr Python Examples

This directory contains complete, runnable examples demonstrating various features and use cases of Zarr Python.

## Directory Structure

Each example is organized in its own subdirectory with the following structure:

```
examples/
├── example_name/
│   ├── README.md          # Documentation for the example
│   └── example_name.py    # Python source code
└── ...
```

## Adding New Examples

To add a new example:

1. Create a new subdirectory: `examples/my_example/`
2. Add your Python code: `examples/my_example/my_example.py`
3. Create documentation: `examples/my_example/README.md`
4. Create a documentation page at `docs/user-guide/examples/my_example.md`. The documentation page should simply link to the `README.md` and the source code, e.g.:

    ````
    # docs/user-guide/examples/my_example.md
    --8<-- "examples/my_example/README.md"

    ## Source Code

    ```python
    --8<-- "examples/my_example/my_example.py"
    ```
    ````
5. Update `mkdocs.yml` to include the new example in the navigation.

### Example README.md Format

Your README.md should include:

- A title (`# Example Name`)
- Description of what the example demonstrates
- Instructions for running the example
