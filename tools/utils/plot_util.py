import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_cmap(n, name='hsv'):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    return plt.cm.get_cmap(name, n)


def smooth_vals(y_vals, alpha):
    y_vals = np.array(y_vals)
    current_y = y_vals[0]
    new_ys = []
    for y in y_vals:
        current_y = current_y * alpha + (1 - alpha) * y
        new_ys.append(current_y)
    return pd.Series(new_ys)
