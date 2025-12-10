# mllogic/double_ml/visualisation.py

from typing import Optional
import matplotlib.pyplot as plt
import pandas as pd


def compute_ate_by_group_did(
    panel: pd.DataFrame,
    control_col: str,
    group_col: str,
    pre_window=(-5, -1),
    post_window=(0, 10),
    price_col: str = "prix_m2_mean",
) -> pd.DataFrame:
    """
    Diff-in-Diff sur la variable price_col (par défaut prix_m2_mean)
    vs une zone contrôle (control_col), agrégé par group_col.

    Retourne :
      - pre_mean, post_mean, ate_pct
      - n_pre / n_post (nombre de mois)
      - n_tx_pre / n_tx_post (nombre de transactions)
      - n_obs (mois totaux) et n_transactions (tx totales)
      - ci_low_pct / ci_high_pct (IC 95 % sur l’ATE)
    """

    df = panel.copy()

    # On garde uniquement ce qui est exploitable
    df = df[
        df[price_col].notna()
        & df[control_col].notna()
        & df["relative_signature_year"].notna()
    ][[group_col, "relative_signature_year", price_col, control_col, "n_transactions"]]

    # Écart en % vs contrôle, mois par mois
    df["diff_pct"] = (df[price_col] / df[control_col] - 1.0) * 100

    pre_min, pre_max = pre_window
    post_min, post_max = post_window

    df_pre = df[
        (df["relative_signature_year"] >= pre_min)
        & (df["relative_signature_year"] <= pre_max)
    ].copy()

    df_post = df[
        (df["relative_signature_year"] >= post_min)
        & (df["relative_signature_year"] <= post_max)
    ].copy()

    # Agrégations pré / post
    agg_pre = (
        df_pre.groupby(group_col)
        .agg(
            pre_mean=("diff_pct", "mean"),
            n_pre=("diff_pct", "size"),
            n_tx_pre=("n_transactions", "sum"),
        )
    )

    agg_post = (
        df_post.groupby(group_col)
        .agg(
            post_mean=("diff_pct", "mean"),
            n_post=("diff_pct", "size"),
            n_tx_post=("n_transactions", "sum"),
        )
    )

    res = agg_pre.join(agg_post, how="inner")  # garde uniquement groupes avec pré & post
    res["ate_pct"] = res["post_mean"] - res["pre_mean"]

    # Construction DiD pour l'IC : pré avec signe -, post avec signe +
    df_pre["signed_diff"] = -df_pre["diff_pct"]
    df_post["signed_diff"] = df_post["diff_pct"]

    df_signed = pd.concat([df_pre, df_post], axis=0)

    agg_ci = (
        df_signed.groupby(group_col)["signed_diff"]
        .agg(
            std_signed=lambda s: s.std(ddof=1),
            n_obs="size",
        )
    )

    res = res.join(agg_ci)

    # IC 95 %
    res["ci_low_pct"] = res["ate_pct"] - 1.96 * res["std_signed"] / np.sqrt(res["n_obs"])
    res["ci_high_pct"] = res["ate_pct"] + 1.96 * res["std_signed"] / np.sqrt(res["n_obs"])

    res["n_transactions"] = res["n_tx_pre"] + res["n_tx_post"]

    res = res.reset_index()

    return res[
        [
            group_col,
            "pre_mean",
            "n_pre",
            "n_tx_pre",
            "post_mean",
            "n_post",
            "n_tx_post",
            "ate_pct",
            "n_obs",
            "n_transactions",
            "ci_low_pct",
            "ci_high_pct",
        ]
    ]
