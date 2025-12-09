# mllogic/double_ml/visualisation.py

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_ate_point(
    ate: float,
    ci_low: float,
    ci_high: float,
    title: str = "ATE (Double ML)",
):
    """
    Plot simple d'un ATE avec intervalle de confiance.
    """
    err_low = ate - ci_low
    err_high = ci_high - ate

    plt.figure(figsize=(4, 5))
    plt.errorbar(
        x=[0],
        y=[ate],
        yerr=[[err_low], [err_high]],
        fmt="o",
        ecolor="black",
        capsize=5,
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.xlim(-1, 1)
    plt.xticks([])
    plt.ylabel("ATE (points de % ou unité de y)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_ate_by_group(
    df: pd.DataFrame,
    group_col: str,
    ate_col: str = "ate",
    ci_low_col: str = "ci_low",
    ci_high_col: str = "ci_high",
    title: Optional[str] = None,
    rotation: int = 45,
):
    """
    Barplot ATE par groupe avec barres d'erreur (IC).

    df doit contenir [group_col, ate_col, ci_low_col, ci_high_col].
    """
    df = df.copy()

    df["err_low"] = df[ate_col] - df[ci_low_col]
    df["err_high"] = df[ci_high_col] - df[ate_col]

    plt.figure(figsize=(8, 4))
    plt.bar(df[group_col], df[ate_col])
    plt.errorbar(
        x=df[group_col],
        y=df[ate_col],
        yerr=[df["err_low"], df["err_high"]],
        fmt="none",
        ecolor="black",
        capsize=3,
    )
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(rotation=rotation)
    plt.ylabel("ATE")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()
