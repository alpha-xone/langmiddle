@echo off
REM Batch script to build and check the package

echo Cleaning previous builds...
if exist dist rmdir /s /q dist
if exist build rmdir /s /q build
if exist langmiddle.egg-info rmdir /s /q langmiddle.egg-info

echo Installing build dependencies...
python -m pip install --upgrade pip
pip install build twine

echo Building package...
python -m build

echo Checking package...
twine check dist/*

echo Build complete! Files are in the dist/ directory.
echo To upload to PyPI, run: twine upload dist/*