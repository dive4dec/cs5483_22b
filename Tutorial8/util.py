import os
import matplotlib.pyplot as plt
import numpy as np


def plot_cluster_regions(X, Y, clusterer, target_names=None, ax=None, N=200):
    """Plot the cluster regions of a clusterer on a 2D dataset.

    Parameters
    ----------
    X: 2D array
        Input features.
    Y: 1D array
        Target values for X.
    clusterer (sklean clusterer):
        A clusterer trained to cluster X.
    target_names: 1D array
        Possible target names.
        Default: None, which means infer from Y.
    ax: axis
        axis to plot the boundaries.
    N: int
        Number of points for each dimension to scan for the decision boundaries.

    Returns
    -------
    axis: axis for the plot of the decision boundaries
    """

    def cat2int(target_array):
        return (
            (np.asarray(target_array).reshape(-1, 1) == target_names.reshape(1, -1))
            * np.arange(len(target_names)).reshape(1, -1)
        ).sum(axis=-1)

    if ax is None:
        ax = plt.gca()

    X_, Y_ = np.asarray(X), np.asarray(Y)

    X_min, X_max = X_.min(axis=0), X_.max(axis=0)
    x1, x2 = np.mgrid[X_min[0]:X_max[0]:N * 1j, X_min[1]:X_max[1]:N * 1j]

    if target_names is None:
        target_names = np.unique(Y_)

    yhat = clusterer.predict(np.c_[x1.ravel(), x2.ravel()]).reshape(x1.shape) 

    ax.contourf(x1, x2, yhat, alpha=0.4, cmap="Set1")
    scatter = ax.scatter(X_[:, 0], X_[:, 1], c = cat2int(Y_), edgecolor='w', s=20, cmap="Set1")
    ax.set_xlim(X_min[0], X_max[0])
    ax.set_ylim(X_min[1], X_max[1])
    ax.add_artist(
        ax.legend(
            scatter.legend_elements()[0],
            target_names,
            loc="upper left",
            title="Classes"))
    return ax
