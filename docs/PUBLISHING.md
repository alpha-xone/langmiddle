# PyPI Publishing Setup Guide

This document explains how to set up and publish the `langmiddle` package to PyPI.

## Prerequisites

1. **PyPI Account**: Create accounts on both [PyPI](https://pypi.org/account/register/) and [TestPyPI](https://test.pypi.org/account/register/)
2. **API Tokens**: Generate API tokens for both PyPI and TestPyPI
3. **GitHub Repository**: Ensure your code is pushed to GitHub

## Setup Steps

### 1. Generate PyPI API Tokens

#### For PyPI (Production):
1. Go to [PyPI Account Settings](https://pypi.org/manage/account/)
2. Scroll to "API tokens" section
3. Click "Add API token"
4. Name: `langmiddle-github-actions`
5. Scope: `Entire account` (or `Project: langmiddle` if package already exists)
6. Copy the generated token (starts with `pypi-`)

#### For TestPyPI (Testing):
1. Go to [TestPyPI Account Settings](https://test.pypi.org/manage/account/)
2. Follow the same steps as above
3. Copy the generated token

### 2. Add Secrets to GitHub Repository

1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Add the following secrets:
   - `PYPI_API_TOKEN`: Your PyPI production token
   - `TEST_PYPI_API_TOKEN`: Your TestPyPI token

### 3. Package Structure

The repository now has the following structure for PyPI publishing:

```
langmiddle/
├── .github/
│   └── workflows/
│       ├── ci.yml              # Continuous Integration
│       └── publish.yml         # PyPI Publishing
├── langmiddle/                 # Main package directory
│   ├── __init__.py            # Package initialization
│   └── py.typed               # Type checking marker
├── tests/                     # Test directory
│   ├── __init__.py
│   ├── conftest.py
│   └── test_basic.py
├── scripts/                   # Utility scripts
│   ├── publish.py             # Manual publishing script
│   ├── test_install.py        # Installation test script
│   ├── build.bat              # Windows build script
│   └── build.sh               # Unix build script
├── pyproject.toml             # Modern Python packaging configuration
├── requirements.txt           # Production dependencies
├── requirements-dev.txt       # Development dependencies
├── MANIFEST.in               # Additional files to include
├── README.md                 # Project documentation
├── LICENSE                   # License file
└── .gitignore               # Git ignore rules
```

## Publishing Methods

### Method 1: Automatic Publishing via GitHub Releases (Recommended)

1. **Create a Release**:
   ```bash
   git tag v0.1.0
   git push origin v0.1.0
   ```

2. **Create GitHub Release**:
   - Go to your repository on GitHub
   - Click "Releases" → "Create a new release"
   - Choose the tag you just created (`v0.1.0`)
   - Fill in release title and description
   - Click "Publish release"

3. **Automatic Publishing**:
   - The GitHub Action will automatically trigger
   - It will build and publish to PyPI
   - Check the "Actions" tab for progress

### Method 2: Manual Publishing via GitHub Actions

1. Go to your repository's "Actions" tab
2. Click on "Publish to PyPI" workflow
3. Click "Run workflow"
4. Choose the environment:
   - `pypi` for production PyPI
   - `testpypi` for testing

### Method 3: Local Publishing

#### Using the Python Script:
```bash
cd langmiddle
python scripts/publish.py
```

#### Using Build Scripts:

**Windows:**
```cmd
scripts\build.bat
twine upload dist/*
```

**Unix/Linux/macOS:**
```bash
chmod +x scripts/build.sh
./scripts/build.sh
twine upload dist/*
```

#### Manual Commands:
```bash
# Clean previous builds
rm -rf dist/ build/ *.egg-info/

# Install build tools
pip install build twine

# Build package
python -m build

# Check package
twine check dist/*

# Upload to TestPyPI (for testing)
twine upload --repository testpypi dist/*

# Upload to PyPI (production)
twine upload dist/*
```

## Testing the Installation

After publishing, test the installation:

```bash
# Test installation from PyPI
pip install langmiddle

# Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ langmiddle

# Run the automated test script
python scripts/test_install.py
```

## Version Management

Update the version in `langmiddle/__init__.py`:

```python
__version__ = "0.1.1"  # Update this
```

The version is also specified in `pyproject.toml` and should be kept in sync.

## CI/CD Pipeline

The repository includes:

1. **Continuous Integration** (`.github/workflows/ci.yml`):
   - Runs on every push and pull request
   - Tests across multiple Python versions (3.10-3.13)
   - Tests on multiple operating systems (Ubuntu, Windows, macOS)
   - Runs linting, formatting, and type checking

2. **Publishing Workflow** (`.github/workflows/publish.yml`):
   - Triggered by GitHub releases
   - Can be manually triggered with environment selection
   - Builds and publishes to PyPI/TestPyPI

## Package Configuration

Key files for PyPI publishing:

- **`pyproject.toml`**: Modern Python packaging configuration
- **`MANIFEST.in`**: Specifies additional files to include
- **`langmiddle/__init__.py`**: Package metadata and version
- **`requirements.txt`**: Production dependencies

## Troubleshooting

### Common Issues:

1. **Authentication Errors**: Verify API tokens are correctly set in GitHub secrets
2. **Version Conflicts**: Ensure you're incrementing the version number
3. **Build Failures**: Check that all dependencies are properly specified
4. **Import Errors**: Verify package structure and `__init__.py` files

### Useful Commands:

```bash
# Check package metadata
python -m build
twine check dist/*

# View package contents
tar -tf dist/langmiddle-*.tar.gz

# Test local installation
pip install -e .
python -c "import langmiddle; print(langmiddle.__version__)"
```

## Next Steps

1. **Add More Functionality**: Implement your middleware components
2. **Write Tests**: Add comprehensive tests for your code
3. **Documentation**: Consider adding Sphinx documentation
4. **Continuous Deployment**: The setup is ready for automatic publishing

## Resources

- [Python Packaging Guide](https://packaging.python.org/)
- [PyPI Publishing Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
- [GitHub Actions for Python](https://docs.github.com/en/actions/automating-builds-and-tests/building-and-testing-python)