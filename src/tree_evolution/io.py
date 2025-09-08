import pickle
import tempfile
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import papermill as pm
import torch


def store(path, data):
    with open(path, "wb") as file:
        pickle.dump(data, file)


def load(path):
    with open(path, "rb") as file:
        return pickle.load(file)


@contextmanager
def tmp_output_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "result"


def run_notebook(notebook, output, params):
    with tmp_output_file() as out_path:
        pm.execute_notebook(
            notebook,
            output,
            parameters={"OUTPUT_PATH": str(out_path), **params},
            progress_bar=False,
        )
        return load(out_path)


def plot_activations(activations, bounds=(-1, 1), n=100):
    x0, x1 = bounds
    x = torch.linspace(x0, x1, n, requires_grad=False)

    fig, axs = plt.subplots(1, len(activations), figsize=(12, 4))
    for mod, ax in zip(activations, axs, strict=True):
        y = mod(x)
        ax.plot(x.numpy(), y.detach().numpy())

    plt.show()
    plt.close(fig)
