#!/usr/bin/env python3
"""Script to test package installation from PyPI"""

import subprocess
import sys
import tempfile
import os


def run_command(cmd, cwd=None):
    """Run a command and return result"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    print(f"Return code: {result.returncode}")
    if result.stdout:
        print(f"STDOUT: {result.stdout}")
    if result.stderr:
        print(f"STDERR: {result.stderr}")
    return result


def main():
    """Test package installation"""
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Testing in temporary directory: {temp_dir}")

        # Create a virtual environment
        venv_dir = os.path.join(temp_dir, "test_env")
        result = run_command(f"python -m venv {venv_dir}")
        if result.returncode != 0:
            print("Failed to create virtual environment")
            sys.exit(1)

        # Determine python executable path
        if sys.platform == "win32":
            python_exe = os.path.join(venv_dir, "Scripts", "python.exe")
            pip_exe = os.path.join(venv_dir, "Scripts", "pip.exe")
        else:
            python_exe = os.path.join(venv_dir, "bin", "python")
            pip_exe = os.path.join(venv_dir, "bin", "pip")

        # Install the package from PyPI
        result = run_command(f'"{pip_exe}" install langmiddle')
        if result.returncode != 0:
            print("Failed to install package from PyPI")
            sys.exit(1)

        # Test importing the package
        test_script = '''
import langmiddle
print(f"Successfully imported langmiddle version {langmiddle.__version__}")
print(f"Author: {langmiddle.__author__}")
print(f"Email: {langmiddle.__email__}")
'''

        result = run_command(f'"{python_exe}" -c "{test_script}"')
        if result.returncode != 0:
            print("Failed to import and test package")
            sys.exit(1)

        print("Package installation and import test successful!")


if __name__ == "__main__":
    main()
