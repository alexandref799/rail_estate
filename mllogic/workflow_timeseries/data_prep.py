from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from .config import Config


def group_by(
    df: pd.DataFrame,
    colonne_date: str,
    colonne_gare: str,
    colonnes_numeriques: list,
    colonnes_categorielles: list,
) -> pd.DataFrame:
    df = df.copy()
    df[colonne_date] = pd.to_datetime(df[colonne_date])
    df["Annee_Mois"] = df[colonne_date].dt.to_period("M")

    # Reference category (global mode) for each categorical column to compute shares
    reference_cat = {col: df[col].mode()[0] for col in colonnes_categorielles}

    df_agg = (
        df.groupby(["Annee_Mois", colonne_gare])
        .agg(
            n_transactions=("Annee_Mois", "size"),
            **{col: (col, "mean") for col in colonnes_numeriques},
            **{
                f"{col}_share": (
                    col,
                    lambda x, ref=reference_cat[col]: np.mean(x == ref)
                    if len(x) > 0
                    else np.nan,
                )
                for col in colonnes_categorielles
            },
        )
        .reset_index()
    )

    nb_mois_total = df_agg["Annee_Mois"].nunique()
    gares_completes = (
        df_agg.groupby(colonne_gare)["Annee_Mois"]
        .nunique()
        .loc[lambda x: x == nb_mois_total]
        .index
    )
    df_agg = df_agg[df_agg[colonne_gare].isin(gares_completes)]

    df_agg["date"] = df_agg["Annee_Mois"].dt.to_timestamp()
    df_agg = df_agg.drop(columns=["Annee_Mois"]).set_index("date").sort_index()

    cat_share_cols = [f"{col}_share" for col in colonnes_categorielles]
    colonnes_a_garder = (
        [colonne_gare, "n_transactions"]
        + colonnes_numeriques
        + cat_share_cols
    )
    return df_agg[colonnes_a_garder].dropna()


def create_sequences(X, y=None, n_steps=12):
    Xs, ys = [], []
    for i in range(len(X) - n_steps):
        Xs.append(X[i : i + n_steps])
        if y is not None:
            ys.append(y[i + n_steps])
    return np.array(Xs), (np.array(ys) if y is not None else None)


def prepare_data(cfg: Config, gare: str):
    df_new = pd.read_csv(cfg.data_path, sep=";")
    if "Unnamed: 0" in df_new.columns:
        df_new = df_new.drop(columns="Unnamed: 0")

    df_group = group_by(
        df_new,
        cfg.colonne_date,
        cfg.colonne_gare,
        cfg.colonnes_numeriques,
        cfg.colonnes_categorielles,
    )

    df_gare = df_group[df_group[cfg.colonne_gare] == gare].sort_index()
    df_train = df_gare.iloc[: -cfg.forecast_horizon]
    df_test = df_gare.iloc[-cfg.forecast_horizon :]

    # After aggregation, categorical columns became share columns
    categorical_features = []
    share_features = [f"{col}_share" for col in cfg.colonnes_categorielles]
    numerical_features = (
        [
            col
            for col in df_gare.columns
            if col not in share_features + [cfg.target, cfg.colonne_gare]
        ]
        + share_features
    )
    # Deduplicate in case of overlap
    numerical_features = list(dict.fromkeys(numerical_features))

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), numerical_features),
            (
                "cat",
                OneHotEncoder(
                    sparse_output=False, drop="if_binary", handle_unknown="ignore"
                ),
                categorical_features,
            ),
        ]
    )

    X_train = preprocessor.fit_transform(df_train)
    X_test = preprocessor.transform(df_test)

    y_scaler = MinMaxScaler()
    y_train = y_scaler.fit_transform(df_train[[cfg.target]])
    y_test = y_scaler.transform(df_test[[cfg.target]])

    X_train_seq, y_train_seq = create_sequences(X_train, y_train, cfg.n_steps)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, cfg.n_steps)

    return {
        "X_train_seq": X_train_seq,
        "y_train_seq": y_train_seq,
        "X_test_seq": X_test_seq,
        "y_test_seq": y_test_seq,
        "y_scaler": y_scaler,
        "df_test": df_test,
        "preprocessor": preprocessor,
    }
