import dill
import matplotlib.pyplot as plt
import numpy as np
import torch


def store(path, data):
    with open(path, "wb") as file:
        dill.dump(data, file)


def load(path):
    with open(path, "rb") as file:
        return dill.load(file)


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
