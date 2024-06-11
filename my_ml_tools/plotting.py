import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
import ipywidgets as widgets
from IPython.display import display
from ipywidgets import interact

def generate_meshgrid(X: np.ndarray, resolution: int = 100) -> list[np.ndarray,np.ndarray]:
    """Generates a mesh grid for plotting decision boundaries."""
    xmin, xmax = X[:, 0].min() - 0.3, X[:, 0].max() + 0.3
    ymin, ymax = X[:, 1].min() - 0.3, X[:, 1].max() + 0.3
    return np.meshgrid(
        np.linspace(xmin, xmax, resolution),
        np.linspace(ymin, ymax, resolution)
    )

def plot_decision_boundary_common(X, y, pred, cmap, alpha, figsize, labels, markers, title, xlabel, ylabel, legend_loc, confidence=None) -> None:
    """Common plotting function for decision boundaries."""
    plt.figure(figsize=figsize)
    plt.contourf(*pred, cmap=cmap, alpha=alpha)

    classes = np.unique(y)
    colors = cm.get_cmap(cmap)(np.linspace(0, 1, len(classes)))
    
    for i, cls in enumerate(classes):
        plt.scatter(X[y == cls, 0], X[y == cls, 1], color=colors[i], label=labels[int(cls)] if labels else cls, marker=markers[i % len(markers)])
    
    if confidence is not None:
        plt.contour(*pred, levels=confidence, colors='k', linestyles='dashed')

    plt.legend(loc=legend_loc)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def plot_decision_boundary(X, y, model, *, cmap='turbo', alpha=0.5, figsize=(10, 7), labels=[], markers=['o', 's', 'D', '^', 'v'],
                           title='Decision Boundary', xlabel='Feature 1', ylabel='Feature 2', legend_loc='best', resolution=100,
                           show=True, confidence=None, verbose=False) -> None:
    """Plots the decision boundary of a given model."""
    mesh_x, mesh_y = generate_meshgrid(X, resolution)
    x_stacked = np.c_[mesh_x.ravel(), mesh_y.ravel()]
    pred = model.predict(x_stacked).reshape(mesh_x.shape)
    
    if verbose:
        print(f"Mesh grid created with resolution {resolution}x{resolution}.")
    
    if show:
        plot_decision_boundary_common(X, y.squeeze(), (mesh_x, mesh_y, pred), cmap, alpha, figsize, labels, markers, title, xlabel, ylabel, legend_loc, confidence)

def plot_decision_boundary_torch(X, y, model, *, cmap='turbo', alpha=0.5, figsize=(10, 7), labels=[], markers=['o', 's', 'D', '^', 'v'],
                                 title='Decision Boundary', xlabel='Feature 1', ylabel='Feature 2', legend_loc='best', resolution=100,
                                 show=True, confidence=None, verbose=False) -> None:
    """Plots the decision boundary of a given PyTorch model."""
    import torch
    model = model.cpu()
    X = X.cpu().numpy()
    y = y.cpu().numpy().squeeze()
    mesh_x, mesh_y = generate_meshgrid(X, resolution)
    x_stacked = np.c_[mesh_x.ravel(), mesh_y.ravel()]
    tensor_x_stacked = torch.tensor(x_stacked).float()
    pred = model(tensor_x_stacked).detach().argmax(axis=1).reshape(mesh_x.shape)
    
    if verbose:
        print(f"Mesh grid created with resolution {resolution}x{resolution}.")
    
    if show:
        plot_decision_boundary_common(X, y, (mesh_x, mesh_y, pred), cmap, alpha, figsize, labels, markers, title, xlabel, ylabel, legend_loc, confidence)

def plot_decision_boundary_tensorflow(X, y, model, *, cmap='turbo', alpha=0.5, figsize=(10, 7), labels=[], markers=['o', 's', 'D', '^', 'v'],
                                      title='Decision Boundary', xlabel='Feature 1', ylabel='Feature 2', legend_loc='best', resolution=100,
                                      show=True, confidence=None, verbose=False) -> None:
    """Plots the decision boundary of a given TensorFlow model."""
    mesh_x, mesh_y = generate_meshgrid(X, resolution)
    x_stacked = np.c_[mesh_x.ravel(), mesh_y.ravel()]
    pred = model.predict(x_stacked).argmax(axis=1).reshape(mesh_x.shape)
    
    if verbose:
        print(f"Mesh grid created with resolution {resolution}x{resolution}.")
    
    if show:
        plot_decision_boundary_common(X, y, (mesh_x, mesh_y, pred), cmap, alpha, figsize, labels, markers, title, xlabel, ylabel, legend_loc, confidence)

def plot_decision_boundary_models(X, y, model_list, *, cmap='turbo', alpha=0.5, figsize=(16, 7), labels=[], markers=['o', 's', 'D', '^', 'v'],
                                  title='Decision Boundary', xlabel='Feature 1', ylabel='Feature 2', legend_loc='lower right', resolution=100,
                                  save_path=None, show=True, confidence=None, verbose=False) -> None:
    """Plots the decision boundaries of a list of models."""
    from matplotlib.gridspec import GridSpec

    for model in model_list:
        model.fit(X, y)
    
    length = len(model_list)
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, length) if length <= 5 else GridSpec(1 + length // 5, 5)
    
    mesh_x, mesh_y = generate_meshgrid(X, resolution)
    x_stacked = np.c_[mesh_x.ravel(), mesh_y.ravel()]
    classes = np.unique(y)
    colors = cm.get_cmap(cmap)(np.linspace(0, 1, len(classes)))

    if verbose:
        print(f"Mesh grid created with resolution {resolution}x{resolution}.")
    
    for i, model in enumerate(model_list):
        ax = fig.add_subplot(gs[i // 5, i % 5])
        pred = model.predict(x_stacked).reshape(mesh_x.shape)
        ax.contourf(mesh_x, mesh_y, pred, cmap=cmap, alpha=alpha)
        
        for j, cls in enumerate(classes):
            ax.scatter(X[y == cls, 0], X[y == cls, 1], color=colors[j], marker=markers[j % len(markers)])
        
        ax.set_title(f'Model {i+1}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    
    fig.legend(labels if labels else classes, loc=legend_loc)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    if show:
        plt.show()



def interactive_decision_boundary(X, y, model, cmap_options=None) -> None:
    """Creates interactive widgets to plot decision boundaries."""
    cmap_options = cmap_options or ['turbo', 'viridis', 'plasma', 'inferno', 'magma', 'cividis','jet','flag', 'prism', 'ocean', 'gist_earth', 'terrain',
                                    'gist_stern', 'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'gist_rainbow', 'rainbow', 'nipy_spectral', 'gist_ncar']

    cmap_selector = widgets.Dropdown(options=cmap_options, value='turbo', description='Colormap:')
    alpha_slider = widgets.FloatSlider(value=0.5, min=0, max=1, step=0.05, description='Alpha:')
    resolution_slider = widgets.IntSlider(value=100, min=10, max=500, step=10, description='Resolution:')
    title_text = widgets.Text(value='Decision Boundary', description='Title:')
    xlabel_text = widgets.Text(value='Feature 1', description='X Label:')
    ylabel_text = widgets.Text(value='Feature 2', description='Y Label:')
    legend_loc_selector = widgets.Dropdown(options=['best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'], value='best', description='Legend Loc:')
    verbose_checkbox = widgets.Checkbox(value=False, description='Verbose')

    ui = widgets.VBox([cmap_selector, alpha_slider, resolution_slider, title_text, xlabel_text, ylabel_text, legend_loc_selector, verbose_checkbox])

    def update_plot(cmap, alpha, resolution, title, xlabel, ylabel, legend_loc, verbose):
        plot_decision_boundary(X, y, model, cmap=cmap, alpha=alpha, resolution=resolution, title=title, xlabel=xlabel, ylabel=ylabel, legend_loc=legend_loc, verbose=verbose)
    
    out = widgets.interactive_output(update_plot, {'cmap': cmap_selector, 'alpha': alpha_slider, 'resolution': resolution_slider, 'title': title_text, 'xlabel': xlabel_text, 'ylabel': ylabel_text, 'legend_loc': legend_loc_selector, 'verbose': verbose_checkbox})

    display(ui, out)

def plot_confusion_matrix(y_true, y_pred, classes=None, normalize=False, cmap='Blues', title='Confusion Matrix', figsize=(10, 7), annot=True, fmt='.2f') -> None:
    """
    Plots a confusion matrix with enhancements for better visualization.

    Parameters:
    y_true : array-like of shape (n_samples,)
        True labels.
    y_pred : array-like of shape (n_samples,)
        Predicted labels.
    classes : list of str, optional
        Class names for labels. If not provided, unique labels will be used.
    normalize : bool, optional
        Whether to normalize the confusion matrix.
    cmap : str, optional
        Colormap for the plot.
    title : str, optional
        Title for the plot.
    figsize : tuple, optional
        Size of the figure.
    annot : bool, optional
        Whether to annotate the cells with the counts or proportions.
    fmt : str, optional
        String format for the annotations.

    Returns:
    None
    """
    from sklearn.utils.multiclass import unique_labels
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Use unique labels if classes are not provided
    if classes is None:
        classes = unique_labels(y_true, y_pred)

    # Create the plot
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt=fmt if normalize else 'd', cmap=cmap, xticklabels=classes, yticklabels=classes)

    # Add title and labels
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Add color bar
    # plt.colorbar()

    # Adjust layout for better fit
    plt.tight_layout()
    plt.show()
