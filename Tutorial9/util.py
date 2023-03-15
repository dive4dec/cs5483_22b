import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram


def plot_dendrogram(model, ax=None, **kwargs):
    """Plot the dendrogram of a clusterer on a 2D dataset using
    scipy.cluster.hierarchy.dendrogram.

    Parameters
    ----------
    model: fitted sklearn clusterer
        An AgglomerativeClustering object with cluster distances computed,
        e.g., by setting the parameter compute_distances=True and applying fit
        method to samples.
    ax: axis
        Axis to plot the dendrogram.
    **kwargs: keyword arguments
        Additional parameters for scipy.cluster.hierarchy.dendrogram

    Returns
    -------
    axis:
        axis for the plot of the decision boundaries

    See also
    --------
    scipy.cluster.hierarchy.dendrogram

    Reference
    ---------
    https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
    """
    if ax is None:
        ax = plt.gca()

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, ax=ax, **kwargs)
    return ax