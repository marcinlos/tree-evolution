# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% editable=true slideshow={"slide_type": ""}
import torch
from torch import nn

from tree_evolution.io import plot_activations, store
from tree_evolution.nn import decode_activations

# %% editable=true slideshow={"slide_type": ""} tags=["parameters"]
N = 100
NEURONS_PER_LAYER = 512
EPOCHS = 200

ACTIVATIONS = None
OUTPUT_PATH = None

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
    def __init__(self, activations, size=512):
        super().__init__()

        modules = [nn.Linear(2, size)]

        for act in activations:
            modules.append(act)
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

# %% editable=true slideshow={"slide_type": ""}
activations = decode_activations(ACTIVATIONS)
pinn = PINN(activations, size=NEURONS_PER_LAYER).to(DEVICE)
print(pinn)

# %%
plot_activations(activations)


# %% editable=true slideshow={"slide_type": ""}
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
loss_fn = LossFunction(fun, N)


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
            plot_activations(activations)

    return log


# %% editable=true slideshow={"slide_type": ""}
log = train(pinn, EPOCHS, verbose=True)

# %%
store(OUTPUT_PATH, log)
