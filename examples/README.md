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
4. Run the documentation generator: `python3 scripts/generate_examples_docs.py`

The documentation generator will automatically:
- Create a documentation page at `docs/user-guide/examples/my_example.md`
- Update `mkdocs.yml` to include the new example in the navigation

### Example README.md Format

Your README.md should include:

- A title (# Example Name)
- Description of what the example demonstrates
- Instructions for running the example

**Note:** The source code will be automatically appended to the documentation when you run the generator script.
