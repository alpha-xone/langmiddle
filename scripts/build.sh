#!/bin/bash
# Shell script to build and check the package

echo "Cleaning previous builds..."
rm -rf dist/
rm -rf build/
rm -rf *.egg-info/

echo "Installing build dependencies..."
python -m pip install --upgrade pip
pip install build twine

echo "Building package..."
python -m build

echo "Checking package..."
twine check dist/*

echo "Build complete! Files are in the dist/ directory."
echo "To upload to PyPI, run: twine upload dist/*"