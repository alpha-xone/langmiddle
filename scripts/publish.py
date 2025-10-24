#!/usr/bin/env python3
"""Script to publish package to PyPI"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd):
    """Run a command and check for errors"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running: {cmd}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        sys.exit(1)
    return result.stdout


def main():
    """Main publishing function"""
    # Ensure we're in the right directory
    if not Path("pyproject.toml").exists():
        print("Error: pyproject.toml not found. Run from project root.")
        sys.exit(1)

    print("Cleaning previous builds...")
    run_command("python -c \"import shutil; shutil.rmtree('dist', ignore_errors=True)\"")
    run_command("python -c \"import shutil; shutil.rmtree('build', ignore_errors=True)\"")

    print("Building package...")
    run_command("python -m build")

    print("Checking package...")
    run_command("twine check dist/*")

    # Ask for confirmation before uploading
    response = input("Upload to PyPI? (y/N): ").strip().lower()
    if response not in ['y', 'yes']:
        print("Upload cancelled.")
        sys.exit(0)

    print("Uploading to PyPI...")
    run_command("twine upload dist/*")

    print("Successfully published to PyPI!")


if __name__ == "__main__":
    main()
