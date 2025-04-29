import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

def plot_training_history(history, metrics=('loss', 'mean_absolute_error')):
    """
    plot training & validation curves for inputted metrics
    
    input:
        history: keras history object ??returned by model.fit??
        metrics: metric names to plot     default tuple ('loss','mean_absolute_error')
    """
    epochs = range(1, len(history.history[next(iter(history.history))]) + 1)
    plt.figure(figsize=(8, 5))
    for m in metrics:
        train_key = m
        val_key = f'val_{m}'
        if train_key in history.history:
            plt.plot(epochs, history.history[train_key], label=f'Train {m}')
        if val_key in history.history:
            plt.plot(epochs, history.history[val_key], '--', label=f'Value {m}')
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.title('training and validation metrics')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def scatter_pred_vs_true(y_true, y_pred):
    """
    scatter plot of pred v. true values
    
    inputs:
        y_true: array of true populations per node [something shape (N,) or (N,1)]
        y_pred: array of prediction outputs, same shape as y_true
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], 'k--', lw=1)
    plt.xlabel('true ridership')
    plt.ylabel('[redicted ridership')
    plt.title('predicted v. true scatter')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_error_distribution(y_true, y_pred, bins=50):
    """
    histogram of pred errors
    
    inputs:
        y_true: array-like of true targets
        y_pred: array-like of predictions
        bins: number of histogram bins
    """
    errors = np.array(y_pred).flatten() - np.array(y_true).flatten()
    plt.figure(figsize=(6, 4))
    plt.hist(errors, bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel('Prediction Error')
    plt.ylabel('Count')
    plt.title('Error Distribution')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_node_time_series(node_ind, true_series, pred_series, timestamps=None):
    """
    plot true vs pred time series for one node.
    
    param:
        node_ind: identifier for the node (for title)
        true_series: shape (T,)
        pred_series: shape (T,)
        timestamps: length T timestamps for x-axis
    """
    true_series = np.array(true_series)
    pred_series = np.array(pred_series)
    if timestamps is None:
        timestamps = np.arange(len(true_series))
    
    plt.figure(figsize=(8, 4))
    plt.plot(timestamps, true_series, label='True', marker='o')
    plt.plot(timestamps, pred_series, label='Predicted', marker='x')
    plt.xlabel('Time Step')
    plt.ylabel('Ridership')
    plt.title(f'Node {node_ind} Time Series')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_pearson_over_windows(window_timestamps, y_true_list, y_pred_list):
    """
    compute and plot Pearson r for each evaluation window.

    inputs:
        window_timestamps: list of time identifiers for each window
        y_true_list: list of arrays of true values per window
        y_pred_list: list of arrays of preds per window
    """
    rs = []
    for yt, yp in zip(y_true_list, y_pred_list):
        r, _ = pearsonr(np.ravel(yt), np.ravel(yp))
        rs.append(r)
    plt.figure(figsize=(8, 4))
    plt.plot(window_timestamps, rs, marker='o')
    plt.xlabel('Window')
    plt.ylabel('Pearson r')
    plt.title('Pearson Correlation per Window')
    plt.grid(True)
    plt.tight_layout()
    plt.show()