# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# Execute this line first on Google Colab
# # %pip install "tree-evolution[gpu] @ git+https://github.com/marcinlos/tree-evolution"

# %% editable=true slideshow={"slide_type": ""}
from pathlib import Path

import leap_ec.problem
import numpy as np
import torch

from tree_evolution.evolution import Evolution, ForestRepresentation, forest_id
from tree_evolution.io import store
from tree_evolution.nb import run_notebook
from tree_evolution.op import OperatorRegister
from tree_evolution.tree import to_dict

# %% editable=true slideshow={"slide_type": ""}
NOTEBOOK = "problems/PINN_Laplace_Gram.ipynb"
SUFFIX = "EJ"
PARAMS = {
    "EPOCHS": 10_000,
    "EXAMPLE": 3,
    "EPSILON": 0.1,
}

# Evolutionary search parameters
COUNT = 2
DEPTH = 2
GENERATIONS = 10
POPULATION_SIZE = 5


# %%
ops = OperatorRegister()


# %% jupyter={"source_hidden": true}
@ops.unary
def zero(x):
    return torch.zeros_like(x)


@ops.unary
def const(x, c):
    return c * torch.ones_like(x)


@ops.unary
def scaled(x, a):
    return a * x


@ops.unary
def abs(x, a):
    return a * torch.abs(x)


@ops.unary
def square(x, a):
    return a * x**2


@ops.unary
def cube(x, a):
    return a * x**3


@ops.unary
def sqrt(x, a):
    return a * torch.sqrt(x)


@ops.unary
def exp(x, a, b):
    return a * torch.exp(b * x)


@ops.unary
def exp2(x, a, b):
    return a * torch.exp(b * x**2)


@ops.unary
def logabs(x, a, b, c):
    return a * torch.log(torch.abs(b + c * x))


@ops.unary
def logexp(x, a, b, c):
    return a * torch.log(torch.abs(b + torch.exp(c * x)))


@ops.unary
def sin(x, a, b):
    return a * torch.sin(b * x)


@ops.unary
def sinh(x, a, b):
    return a * torch.sinh(b * x)


@ops.unary
def asinh(x, a, b):
    return a * torch.asinh(b * x)


@ops.unary
def cos(x, a, b):
    return a * torch.cos(b * x)


@ops.unary
def cosh(x, a, b):
    return a * torch.cosh(b * x)


@ops.unary
def tanh(x, a, b):
    return a * torch.tanh(b * x)


@ops.unary
def atanh(x, a, b):
    return a * torch.atanh(torch.clamp(b * x, -1, 1))


@ops.unary
def max0(x, a, b):
    return a * torch.maximum(b * x, torch.tensor(0.0))


@ops.unary
def min0(x, a, b):
    return a * torch.minimum(b * x, torch.tensor(0.0))


@ops.unary
def erf(x, a, b):
    return a * torch.erf(b * x)


@ops.unary
def sinc(x, a, b):
    return a * torch.sinc(b * x)


@ops.binary
def sum(x, y, a, b):
    return a * x + b * y


@ops.binary
def product(x, y, a):
    return a * x * y


@ops.binary
def quot(x, y, a, b):
    return x / (a * y + b)


@ops.binary
def min(x, y, a, b):
    return torch.minimum(a * x, b * y)


@ops.binary
def max(x, y, a, b):
    return torch.maximum(a * x, b * y)


# %%
store("operators", ops)


# %% jupyter={"source_hidden": true}
def make_problem(notebook, suffix, params, out_dir=Path("out")):
    input_file = Path(notebook)

    def fitness(forest):
        activations = [["expr", to_dict(e)] for e in forest]
        fid = forest_id(forest)
        output_file = out_dir / f"{input_file.stem}_{suffix}_{fid}.ipynb"

        losses = run_notebook(
            input_file,
            output_file,
            {"ACTIVATIONS": activations, "OPERATORS": "operators", **params},
            output_html=True,
        )
        err = np.array(losses)
        return np.min(err)

    return leap_ec.problem.FunctionProblem(fitness, maximize=False)


# %%
problem = make_problem(NOTEBOOK, suffix=SUFFIX, params=PARAMS)

# %%
forest = ForestRepresentation(
    count=COUNT,
    depth=DEPTH,
    unary=ops.unary_operators,
    binary=ops.binary_operators,
)


# %%
evolution = Evolution(problem, POPULATION_SIZE, forest)
evolution.print_population()

# %% editable=true slideshow={"slide_type": ""}
while evolution.generation < GENERATIONS:
    evolution.step()
    if evolution.generation % 1 == 0:
        evolution.print_population()
        print()
