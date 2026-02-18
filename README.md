Mixed-Integer Optimization Toolbox for Space Logistics and Mission Planning in Python

# Installation

## Prerequisites

To build the environment and use the code in this repository, you will need to have the following installed:

- `poetry` for dependency management (see `docs/poetry_installation.md`)
- optimization solvers of your choice (Gurobi, BARON, CPLEX, IPOPT, SCIP, etc. See `docs/ipopt_installation.md` for IPOPT instructions.)
- anaconda/miniconda (optional but recommended for managing Python environments)

## Dependency Installation

With `conda`, run:

```sh
conda create -n slpy python=3.11.2
conda activate slpy
poetry install
```

If `virtualenv` or similar is preferred, run poetry install in the newly created virtual environment with python version 3.11.2 or later.
Optimization problems are modelled using `pyomo` in the current version.

## Testing

Run tests to ensure that the package works correctly after installation:

```sh
pytest
```

### Legacy version for Linux

On Linux, the legacy version lets you use `pygmo` and its compatible solvers for continuous nonlinear programming (NLP) problems.
To install the dependency for this legacy version, run the poetry command with `-E legacy` flag:

```sh
poetry install -E legacy
```

A separate set of tests for the legacy version can be run with:

```sh
pytest -m legacy
```

This version is unsupported on other platforms.

# Quick Start

1. (When using it for the first time) Set up the ``slpy`` environment by installing the packages.
2. Execute ``src/run.py`` in the ``slpy`` environment (after changing any parameters in it as needed).
3. Check the output .csv file in the `data/opt_results` for results.

```sh
conda activate slpy
python src/run.py
```

# Cite this work

If you use or modify this code base to generate results for a publication, please cite at least one of the corresponding papers listed below:

```bib
@article{isaji_2022_int_space_logistics_decomp,
  title = {Multidisciplinary Design Optimization Approach to Integrated Space Mission Planning and Spacecraft Design},
  author = {Isaji, Masafumi and Takubo, Yuji and Ho, Koki},
  year = {2022},
  month = sep,
  journal = {Journal of Spacecraft and Rockets},
  publisher = {{American Institute of Aeronautics and Astronautics}},
  volume = {59},
  number = {5},
  pages = {1660--1670},
  doi = {10.2514/1.A35284},
  issn = {0022-4650},
}

@inproceedings{isaji_2021_int_space_ascend,
  title = {Multidisciplinary Design Optimization Approach to Integrated Space Mission Planning and Spacecraft Design},
  author = {Isaji, Masafumi and Takubo, Yuji and Ho, Koki},
  year = {2021},
  month = nov,
  booktitle = {AIAA ASCEND 2021},
  publisher = {American Institute of Aeronautics and Astronautics},
  address = {Las Vegas, NV},
  doi = {10.2514/6.2021-4069},
}

@inproceedings{isaji_2024_int_space_uncertain,
  title = {Integrated Space Mission Planning Under Uncertainty via Stochastic and Decomposition-Based Optimization},
  author = {Masafumi Isaji and Koki Ho},
  year = {2024},
  month = jul,
  booktitle = {AIAA Aviation Forum and ASCEND 2024},
  publisher = {American Institute of Aeronautics and Astronautics},
  address = {Las Vegas, NV},
  doi = {10.2514/6.2024-4816},
}
```

# Acknowledgment

This material is based upon work supported by the National Science Foundation under Award No. 1942559.
Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.
