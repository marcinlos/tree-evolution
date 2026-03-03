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

import matplotlib.pyplot as plt
import numpy as np

from tree_evolution.io import run_notebook

# %% jupyter={"source_hidden": true}
activations = [
    [
        "expr",
        {
            "label": "product",
            "children": [
                {
                    "label": "const",
                    "children": [
                        {
                            "label": "quot",
                            "children": [
                                {
                                    "label": "const",
                                    "children": [{"label": "", "children": []}],
                                },
                                {
                                    "label": "abs",
                                    "children": [{"label": "", "children": []}],
                                },
                            ],
                        }
                    ],
                },
                {
                    "label": "square",
                    "children": [
                        {
                            "label": "product",
                            "children": [
                                {
                                    "label": "const",
                                    "children": [{"label": "", "children": []}],
                                },
                                {
                                    "label": "scaled",
                                    "children": [{"label": "", "children": []}],
                                },
                            ],
                        }
                    ],
                },
            ],
        },
    ],
    [
        "expr",
        {
            "label": "product",
            "children": [
                {
                    "label": "exp",
                    "children": [
                        {
                            "label": "product",
                            "children": [
                                {
                                    "label": "abs",
                                    "children": [{"label": "", "children": []}],
                                },
                                {
                                    "label": "exp",
                                    "children": [{"label": "", "children": []}],
                                },
                            ],
                        }
                    ],
                },
                {
                    "label": "abs",
                    "children": [
                        {
                            "label": "product",
                            "children": [
                                {
                                    "label": "scaled",
                                    "children": [{"label": "", "children": []}],
                                },
                                {
                                    "label": "const",
                                    "children": [{"label": "", "children": []}],
                                },
                            ],
                        }
                    ],
                },
            ],
        },
    ],
]

# %%
activations = [
    [
        "expr",
        {
            "label": "sum",
            "children": [
                {
                    "label": "scaled",
                    "children": [
                        {
                            "label": "product",
                            "children": [
                                {
                                    "label": "scaled",
                                    "children": [{"label": "", "children": []}],
                                },
                                {
                                    "label": "abs",
                                    "children": [{"label": "", "children": []}],
                                },
                            ],
                        }
                    ],
                },
                {
                    "label": "exp",
                    "children": [
                        {
                            "label": "product",
                            "children": [
                                {
                                    "label": "const",
                                    "children": [{"label": "", "children": []}],
                                },
                                {
                                    "label": "abs",
                                    "children": [{"label": "", "children": []}],
                                },
                            ],
                        }
                    ],
                },
            ],
        },
    ],
    [
        "expr",
        {
            "label": "quot",
            "children": [
                {
                    "label": "exp",
                    "children": [
                        {
                            "label": "product",
                            "children": [
                                {
                                    "label": "const",
                                    "children": [{"label": "", "children": []}],
                                },
                                {
                                    "label": "square",
                                    "children": [{"label": "", "children": []}],
                                },
                            ],
                        }
                    ],
                },
                {
                    "label": "exp",
                    "children": [
                        {
                            "label": "sum",
                            "children": [
                                {
                                    "label": "const",
                                    "children": [{"label": "", "children": []}],
                                },
                                {
                                    "label": "const",
                                    "children": [{"label": "", "children": []}],
                                },
                            ],
                        }
                    ],
                },
            ],
        },
    ],
]

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
activations_tanh = [["fixed", "tanh"], ["fixed", "tanh"]]
OUTPUT_DIR = "compare"
CASES = [
    # {
    #     "notebook": "problems/problem_simple.ipynb",
    #     "params": {
    #         "N": 50,
    #         "EPOCHS": 200,
    #         "ACTIVATIONS": activations,
    #     },
    #     "name": "case1",
    # },
    # {
    #     "notebook": "problems/problem_simple.ipynb",
    #     "params": {
    #         "N": 100,
    #         "EPOCHS": 100,
    #         "ACTIVATIONS": activations,
    #     },
    #     "name": "case2",
    # },
    {
        "notebook": "problems/PINN_Laplace_Gram.ipynb",
        "params": {
            "EPOCHS": 20_000,
            "EXAMPLE": 3,
            "EPSILON": 0.1,
            "ACTIVATIONS": activations_tanh,
        },
        "name": "laplace_EJ_tanh",
    },
    {
        "notebook": "problems/PINN_Laplace_Gram.ipynb",
        "params": {
            "EPOCHS": 20_000,
            "EXAMPLE": 3,
            "EPSILON": 0.1,
            "ACTIVATIONS": activations,
        },
        "name": "laplace_EJ_tree",
    },
    {
        "notebook": "problems/PINN_Laplace_Gram.ipynb",
        "params": {
            "EPOCHS": 20_000,
            "EXAMPLE": 1,
            "ACTIVATIONS": activations_tanh,
        },
        "name": "laplace_sinsin_tanh",
    },
    {
        "notebook": "problems/PINN_Laplace_Gram.ipynb",
        "params": {
            "EPOCHS": 20_000,
            "EXAMPLE": 1,
            "ACTIVATIONS": activations,
        },
        "name": "laplace_sinsin_tree",
    },
    {
        "notebook": "problems/PINN_Laplace_Gram.ipynb",
        "params": {
            "EPOCHS": 20_000,
            "EXAMPLE": 2,
            "ACTIVATIONS": activations_tanh,
        },
        "name": "laplace_expsin_tanh",
    },
    {
        "notebook": "problems/PINN_Laplace_Gram.ipynb",
        "params": {
            "EPOCHS": 20_000,
            "EXAMPLE": 2,
            "ACTIVATIONS": activations,
        },
        "name": "laplace_expsin_tree",
    },
]


# %%
def run_training(notebook, out, params):
    losses = run_notebook(notebook, out, params)
    return np.array(losses)


# %%
out_dir = Path(OUTPUT_DIR)

for case in CASES:
    notebook = Path(case["notebook"])
    name = case["name"]
    params = case["params"]
    out = out_dir / f"{notebook.stem}_{name}.ipynb"
    loss = run_training(notebook, out, params)
    fig = plt.figure()
    plt.semilogy(loss)
    plt.title(name)
    plt.show()
    plt.close(fig)
