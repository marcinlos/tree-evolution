import os
import pickle
import tempfile
from contextlib import contextmanager
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
    _ensure_dir_exists(output)
    with tmp_output_file() as result_path:
        pm.execute_notebook(
            notebook,
            output,
            parameters={"OUTPUT_PATH": str(result_path), **params},
            progress_bar=False,
        )
        return load(result_path)


def _ensure_dir_exists(output_file: str | os.PathLike) -> None:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_activations(activations, bounds=(-1, 1), n=100):
    x0, x1 = bounds
    device = _extract_device(activations[0])
    x = np.linspace(x0, x1, n)
    x_torch = torch.tensor(x, requires_grad=False).to(device)

    fig, axs = plt.subplots(1, len(activations), figsize=(12, 4))
    for mod, ax in zip(activations, axs, strict=True):
        y = mod(x_torch)
        ax.plot(x, _numpify(y))

    plt.show()
    plt.close(fig)


def _numpify(x):
    return x.detach().cpu().numpy()


def _extract_device(activation):
    return next(activation.parameters()).device
