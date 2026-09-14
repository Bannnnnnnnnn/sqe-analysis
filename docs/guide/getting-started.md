# Getting started

## Installation

Currently, sqe-analysis is available via [the Git repository](https://github.com/RIKEN-SQERT/sqe-analysis).

The Python version supported by sqe-analysis follows the [Scientific Python ecosystem support schedule](https://scientific-python.org/specs/spec-0000/).

### Installing with pyproject.toml

If you are using [uv](https://docs.astral.sh/uv/) and [have a pyproject.toml file](https://docs.astral.sh/uv/guides/projects/) (recommended), install sqe-analysis as follows:
```shell
cd your/project  # folder containing pyproject.toml

# add sqe-analysis as a dependency
uv add "sqe-analysis @ https://github.com/RIKEN-SQERT/sqe-analysis.git"
```
To load the example data, you also need to install the following additional dependencies:
```shell
uv add h5netcdf h5py
```

If you are planning to modify the library, install in editable mode as follows:

```shell
# create a folder for libraries
mkdir lib
cd lib

# clone a local copy of the library
git clone https://github.com/RIKEN-SQERT/sqe-analysis.git

# install editable version
uv add --editable ./sqe-analysis
```


### Installing in a manually managed virtual environment

If you have manually created a virtual environment, use the following commands.
Using uv is recommended, but these should also work with just `pip install` instead of `uv pip install`.
```shell
# activate your virtual environment
source your/virtual/environment/bin/activate

# install
uv pip install "sqe-analysis @ git+https://github.com/RIKEN-SQERT/sqe-analysis.git"

# optional dependencies to load example data
uv pip install h5netcdf h5py
```

If you are planning to modify the library, install in editable mode as follows:
```shell
# create a folder for libraries
mkdir lib
cd lib

# clone a local copy of the library
git clone https://github.com/RIKEN-SQERT/sqe-analysis.git

# install the local copy
uv pip install --editable ./sqe-analysis
```


## Running a simple analysis

**TODO** <!-- would this be just the same as what we already have on the index page? -->

To create additional analysis classes, see [the tutorial](./creating-analyses.md).
