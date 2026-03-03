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
import leap_ec
import leap_ec.probe
import leap_ec.problem
import leap_ec.util
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from tree_evolution.evolution import Evolution, ForestRepresentation
from tree_evolution.nn import ExpressionModule
from tree_evolution.op import binary_operators, operator_map, unary_operators

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # Neural network infrastructure


# %% editable=true slideshow={"slide_type": ""}
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {DEVICE} device")


# %% editable=true slideshow={"slide_type": ""}
class PINN(nn.Module):
    def __init__(self, trees, size=512):
        super().__init__()

        modules = [nn.Linear(2, size)]

        for expr in trees:
            activation = ExpressionModule(expr, operator_map())
            modules.append(activation)
            modules.append(nn.Linear(size, size))

        modules[-1] = nn.Linear(size, 1)

        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        x.requires_grad_()
        return self.layers(x)

    def grad(self, x):
        vals = self(x)
        return torch.autograd.grad(
            vals,
            x,
            grad_outputs=torch.ones_like(vals),
            create_graph=True,
        )[0]


# %% [markdown]
# # Neural network example


# %%
class LossFunction:
    def __init__(self, fun, n):
        self.fun = fun

        x_raw = torch.linspace(0, 1, steps=n)
        y_raw = torch.linspace(0, 1, steps=n)
        x, y = torch.meshgrid(x_raw, y_raw, indexing="ij")

        self.x = x.reshape(-1, 1).requires_grad_(True).to(DEVICE)
        self.y = y.reshape(-1, 1).requires_grad_(True).to(DEVICE)
        self.points = torch.cat([self.x, self.y], dim=1)

    def __call__(self, pinn):
        expected = fun(self.x, self.y)
        actual = pinn(self.points)
        difference = expected - actual
        return difference.pow(2).mean()


# %%
def fun(x, y):
    return torch.exp(x + y)


# %%
loss_fn = LossFunction(fun, 100)


# %%
def train(pinn, steps, verbose=False):
    optimizer = torch.optim.Adamax(pinn.parameters())
    log = []

    for i in range(steps):
        loss = loss_fn(pinn)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
        log.append(loss_val)

        if verbose and i % 10 == 0:
            print(f"Iter {i}: loss = {loss_val}")

    return log


# %% [markdown]
# # Evolutionary algorithm + neural network example

# %%
COUNT = 2
DEPTH = 2
EPOCHS = 100
GENERATIONS = 30
POPULATION_SIZE = 5


# %%
def fun_to_approximate(x, y):
    return torch.exp(x + y)


loss_fn = LossFunction(fun, n=20)


# %%
def fitness(trees):
    pinn = PINN(trees).to(DEVICE)
    losses = train(pinn, EPOCHS)
    err = np.array(losses)
    return np.min(err)


problem = leap_ec.problem.FunctionProblem(fitness, maximize=False)

# %%
forest = ForestRepresentation(
    count=COUNT,
    depth=DEPTH,
    unary=unary_operators(),
    binary=binary_operators(),
)


# %%
evolution = Evolution(problem, POPULATION_SIZE, forest)

# %% editable=true slideshow={"slide_type": ""}
while evolution.generation < GENERATIONS:
    evolution.step()
    if evolution.generation % 1 == 0:
        evolution.print_population()
        print()


# %%
parents = evolution.population
trees = parents[0].genome
tree = trees[0]

mod = ExpressionModule(tree, operator_map())
x = torch.linspace(-1, 1, 100, requires_grad=False)
y = mod(x)
plt.plot(x.numpy(), y.detach().numpy())
