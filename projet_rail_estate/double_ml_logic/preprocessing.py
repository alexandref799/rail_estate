# mllogic/double_ml/preprocessing.py

from typing import Iterable, List, Tuple

import pandas as pd


def prepare_dml_dataframe(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    feature_cols: Iterable[str],
    dropna: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Prépare un DataFrame propre pour le Double ML.

    - garde uniquement outcome, treatment et features
    - cast du traitement en int/bool
    - optionnellement drop des lignes avec NaN

    Retourne :
        df_clean, feature_cols_effective (liste finale des features)
    """
    cols = [outcome_col, treatment_col] + list(feature_cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in df for DML: {missing}")

    dml_df = df[cols].copy()

    # Traitement binaire (0/1)
    dml_df[treatment_col] = dml_df[treatment_col].astype(int)

    if dropna:
        dml_df = dml_df.dropna(axis=0, how="any")

    # recalculer la liste effective des features (au cas où certaines colonnes aient sauté)
    features_effective = [c for c in feature_cols if c in dml_df.columns]

    return dml_df, features_effective
