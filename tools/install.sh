#!/bin/bash
# Install Poetry
pip install poetry
# Install library dependencies with Poetry
poetry install
# Build the library
poetry build
# Install the library
pip install dist/*whl

echo "Installation of flyplot complete"
