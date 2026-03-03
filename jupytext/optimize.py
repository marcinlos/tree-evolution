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

# %% editable=true slideshow={"slide_type": ""}
from pathlib import Path

import leap_ec.problem
import numpy as np

from tree_evolution.evolution import Evolution, ForestRepresentation, forest_id
from tree_evolution.nb import run_notebook
from tree_evolution.op import binary_operators, unary_operators
from tree_evolution.tree import to_dict

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
# NOTEBOOK = "problems/problem_simple.ipynb"
# SUFFIX = "E50_N30"
# PARAMS = {"EPOCHS": 100, "N": 30}

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
def make_problem(notebook, suffix, params, out_dir=Path("out")):
    input_file = Path(notebook)

    def fitness(forest):
        activations = [["expr", to_dict(e)] for e in forest]
        fid = forest_id(forest)
        output_file = out_dir / f"{input_file.stem}_{suffix}_{fid}.ipynb"

        losses = run_notebook(
            input_file,
            output_file,
            {"ACTIVATIONS": activations, **params},
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
    unary=unary_operators(),
    binary=binary_operators(),
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
