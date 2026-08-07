# tree-evolution

This is a library for finding optimal PINN activation functions using evolutionary algorithms.

## Setup

Clone the repository:
```
git clone https://github.com/marcinlos/tree-evolution.git
```
The project is managed using the [uv](https://docs.astral.sh/uv/).
To build the project environment, execute
```
uv sync --extra gpu
```
Finally, generate notebooks from the source files (this part uses [jupytext](https://jupytext.org/)):
```
uv run jupytext --sync $(find jupytext -type f)
```
or
```
just gen-notebooks
```
if you have the [just](https://github.com/casey/just) installed.

## Usage

The usage of the library is demonstrated in the `notebooks/optimize.ipynb` notebook file.
