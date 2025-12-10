import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Sequence, Optional


def plot_predictions(
    dates: Sequence,
    y_true: Sequence,
    y_pred: Sequence,
    title: str = "Pred vs True",
    date_col: str = "date",
):
    """
    Plot model predictions against ground truth, aligned on the same index.

    Args:
        dates: iterable of dates (same length as y_true and y_pred).
        y_true: iterable of true values.
        y_pred: iterable of predicted values.
        title: plot title.
        date_col: name of the date column in the returned DataFrame.
    """
    df_plot = pd.DataFrame({
        date_col: pd.to_datetime(dates),
        "y_true": np.array(y_true).ravel(),
        "y_pred": np.array(y_pred).ravel(),
    }).sort_values(date_col)

    plt.figure(figsize=(10, 4))
    plt.plot(df_plot[date_col], df_plot["y_true"], label="y_true", color="C0")
    plt.plot(df_plot[date_col], df_plot["y_pred"], label="y_pred", color="C1")
    plt.xlabel(date_col)
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return plt.show(),df_plot
