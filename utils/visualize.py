import numpy as np
import matplotlib.pyplot as plt
from utils.graphics import process_img


def plot_ratemap(activations, n_plots=512, width=16, scale=0.6, smooth=True):
    """
    :param activations: (N, nx, nx)
    """
    n_plots = min(len(activations), n_plots)
    height = (n_plots + width - 1) // width

    fig, axs = plt.subplots(figsize=(max(width * scale, 1.0), max(height * scale, 1.0)))
    fig.tight_layout()
    for i in range(n_plots):
        plt.subplot(height, width, i + 1)
        plt.axis('off')
        img = process_img(activations[i, :, :], smooth=smooth)
        plt.imshow(img, cmap='jet')
    return fig


def calc_ratemap(r, x, coords=((0, 2*np.pi), (-2., 2.)), resolutions=(5, 5)):
    """Compute spatial firing fields"""
    B, T, N = r.shape
    dim = len(coords)
    r = r.reshape(-1, N)
    x = x.reshape(-1, dim)

    # sizes and bins
    sizes = tuple([int((coords[i][1] - coords[i][0]) * resolutions[i]) for i in range(dim)])
    bins = np.zeros((B * T, dim), dtype=int)
    for i in range(dim):
        bin = np.digitize(x[:, i], bins=np.linspace(coords[i][0], coords[i][1], sizes[i]))
        bins[:, i] = np.clip(bin, 0, int(sizes[i] - 1))  # clip

    # count
    activations = np.zeros((N,) + sizes)
    counts = np.zeros(sizes)
    for i in range(B * T):
        idx = tuple(bins[i])
        counts[idx] += 1
        # activations[(slice(None),) + idx] += r[i, :]
        activations[:, idx[0], idx[1]] += r[i, :]

    idx = (counts > 0)
    activations[:, idx] /= counts[idx]
    return activations
