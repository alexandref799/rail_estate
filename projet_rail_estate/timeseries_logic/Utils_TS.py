import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
import numpy as np

# Paths and basic configs
csv_path = "df_merge_with_all_gare.csv"  # adapt if needed
col_date = "date"
col_gare = "nom_gare"
target_col = "prix_m2"



cat_cols = ['Nature mutation', 'Type local']

n_steps = 18
horizon = 6

def group_by(
    df: pd.DataFrame,
    colonne_date: str,
    colonne_gare: str,
    colonnes_numeriques: list,
    colonnes_categorielles: list,
):
    num_cols = [
    'Surface reelle bati', 'Nombre pieces principales',
    'distance_gare_km', 'relative_signature', 'relative_opening', 'taux',
    'lat_gare', 'lon_gare', 'variation_mom', 'variation_yoy',
    'Revenu médian 2021', 'Évol. annuelle moy. de la population 2016-2022',
    'Part proprio en rés. principales 2022',
    'Part locataires HLM dans les rés. principales 2022',
    'Part cadres sup. 2022', 'Part logements vacants 2022',
    'Taux de chômage annuel moyen 2024',
    'Part des élèves du privé parmi les élèves du second degré 2024',
    "Nombre d'établissements 2023"
]

    """Aggregate by month/gare and compute mean on numeric + share on key categoricals."""
    df = df.copy()
    df[colonne_date] = pd.to_datetime(df[colonne_date])
    df["Annee_Mois"] = df[colonne_date].dt.to_period("M")

    def _share_of(series: pd.Series, value: str) -> float:
        return (series == value).mean()

    df_agg = (
        df
        .groupby(["Annee_Mois", colonne_gare])
        .agg(
            n_transactions=("Annee_Mois", "size"),
            **{col: (col, "mean") for col in colonnes_numeriques},
            share_appartement=("Type local", lambda x: _share_of(x, "Appartement")),
            share_vente=("Nature mutation", lambda x: _share_of(x, "Vente")),
        )
        .reset_index()
    )

    # Filtre 138 mois
    nb_mois_total = df_agg["Annee_Mois"].nunique()

    gares_completes = (
        df_agg.groupby(colonne_gare)["Annee_Mois"]
        .nunique()
        .loc[lambda x: x == nb_mois_total]
        .index
    )

    df_agg = df_agg[df_agg[colonne_gare].isin(gares_completes)]

    df_agg["date"] = df_agg["Annee_Mois"].dt.to_timestamp()
    df_agg = df_agg.drop(columns=["Annee_Mois"])
    df_agg = df_agg.set_index("date").sort_index()

    colonnes_a_garder = (
        [colonne_gare, "n_transactions"]
        + colonnes_numeriques
        + ["share_appartement", "share_vente"]
    )

    return df_agg[colonnes_a_garder].dropna()





def create_sequences(X, y=None, n_steps=12):
    Xs, ys = [], []
    for i in range(len(X) - n_steps):
        Xs.append(X[i:i+n_steps])
        if y is not None:
            ys.append(y[i+n_steps])
    return np.array(Xs), (np.array(ys) if y is not None else None)

def create_sequences_multi_horizon(X, y, seq_len=12, horizon=12):
    Xs, ys = [], []
    T = len(X)

    for t in range(seq_len, T - horizon + 1):
        x_seq = X[t-seq_len:t]      # (seq_len, n_features)
        y_seq = y[t:t+horizon]      # (horizon,)
        if np.isnan(y_seq).any():
            continue
        Xs.append(x_seq)
        ys.append(y_seq)

    return np.array(Xs), np.array(ys)


@dataclass
class Config:
    # Data
    data_path: Path = Path(
        "/Users/matthieulataste/code/alexandref799/rail_estate/mllogic/df_new_gare.csv"
    )
    colonne_date: str = "date"
    colonne_gare: str = "nom_gare"
    target: str = "prix_m2"

    colonnes_numeriques: List[str] = field(
        default_factory=lambda: [
            "Surface reelle bati",
            "prix_m2",
            "Nombre pieces principales",
            "distance_gare_km",
            "relative_signature",
            "relative_opening",
            "taux",
            'lon',
            'lat',
            'distance_gare_km',
            'nearest_gare',
            'nom_gare',
            'date_signature',
            'date_ouverture',
            'lat_gare',
            'lon_gare',
            'variation_mom',
            'variation_yoy',
            'Revenu médian 2021',
            'Évol. annuelle moy. de la population 2016-2022',
            'Part proprio en rés. principales 2022',
            'Part locataires HLM dans les rés. principales 2022',
            'Part cadres sup. 2022', 'Part logements vacants 2022',
            'Taux de chômage annuel moyen 2024',
            'Part des élèves du privé parmi les élèves du second degré 2024',
            "Nombre d'établissements 2023"
        ]
    )
    colonnes_categorielles: List[str] = field(
        default_factory=lambda: [
            "Nature mutation",
            "Type local",
        ]
    )

    # Time setup
    forecast_horizon: int = 24
    n_steps: int = 12

    # Model hyperparameters
    lstm_units: List[int] = field(default_factory=lambda: [64, 32])
    gru_units: List[int] = field(default_factory=lambda: [64, 32])
    dropout: float = 0.2

    # Training
    epochs: int = 200
    batch_size: int = 32
    validation_split: float = 0.2
    patience: int = 15
    learning_rate: float = 0.001

    # Outputs
    models_dir: Path = Path("mllogic/models")
    outputs_dir: Path = Path("mllogic/outputs")


def split_train_test(
    df: pd.DataFrame,
    cfg: Config,
    test_start: str | pd.Timestamp | None = None,
    test_end: str | pd.Timestamp | None = None,
):
    """
    Split a single-gare dataframe into train/test.

    If test_start/test_end are provided:
      - train: rows strictly before test_start
      - test : rows between [test_start, test_end] inclusive
    Otherwise, fallback to legacy split (last forecast_horizon rows as test).
    """
    df = df.copy()

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        if cfg.colonne_date in df.columns:
            df[cfg.colonne_date] = pd.to_datetime(df[cfg.colonne_date])
            df = df.set_index(cfg.colonne_date)
        else:
            raise ValueError("No datetime index and no colonne_date column to convert.")
    df = df.sort_index()

    if test_start is None or test_end is None:
        if len(df) <= cfg.forecast_horizon:
            raise ValueError("Not enough rows for this gare to create a test set.")
        df_train = df.iloc[: -cfg.forecast_horizon]
        df_test = df.iloc[-cfg.forecast_horizon :]
        return df_train, df_test

    test_start = pd.to_datetime(test_start)
    test_end = pd.to_datetime(test_end)

    train_mask = df.index < test_start
    test_mask = (df.index >= test_start) & (df.index <= test_end)

    df_train = df.loc[train_mask]
    df_test = df.loc[test_mask]

    if df_train.empty or df_test.empty:
        raise ValueError(f"Train size= {len(df_train)}, Test size={len(df_test)} One of the splits is empty; adjust test_start/test_end.")

    return df_train, df_test

# Aggregate by month/gare with shares for categorical

def group_by2(df):

    num_cols = [
    'Surface reelle bati', 'Nombre pieces principales',
    'distance_gare_km', 'relative_signature', 'relative_opening', 'taux',
    'lat_gare', 'lon_gare', 'variation_mom', 'variation_yoy',
    'Revenu médian 2021', 'Évol. annuelle moy. de la population 2016-2022',
    'Part proprio en rés. principales 2022',
    'Part locataires HLM dans les rés. principales 2022',
    'Part cadres sup. 2022', 'Part logements vacants 2022',
    'Taux de chômage annuel moyen 2024',
    'Part des élèves du privé parmi les élèves du second degré 2024',
    "Nombre d'établissements 2023"
]

    df = df.copy()
    df[col_date] = pd.to_datetime(df[col_date])
    df["Annee_Mois"] = df[col_date].dt.to_period("M")

    def _share(series, value):
        return (series == value).mean()

    agg = (
        df
        .groupby(["Annee_Mois", col_gare])
        .agg(
            n_transactions=("Annee_Mois", "size"),
            **{c: (c, "mean") for c in num_cols},
            prix_m2=('prix_m2', "mean"),
            share_appartement=("Type local", lambda x: _share(x, "Appartement")),
            share_vente=("Nature mutation", lambda x: _share(x, "Vente"))
        )
        .reset_index()
        )

    agg["date"] = agg["Annee_Mois"].dt.to_timestamp()
    agg = agg.drop(columns=["Annee_Mois"])
    agg = agg.set_index("date").sort_index()

    keep_cols = [col_gare, "n_transactions", "prix_m2"] + num_cols + ["share_appartement", "share_vente"]
    return agg[keep_cols].dropna()

def make_sequences_nico(df_split, seq_len=n_steps, horizon=horizon):

    num_cols = [
    'Surface reelle bati', 'Nombre pieces principales',
    'distance_gare_km', 'relative_signature', 'relative_opening', 'taux',
    'lat_gare', 'lon_gare', 'variation_mom', 'variation_yoy',
    'Revenu médian 2021', 'Évol. annuelle moy. de la population 2016-2022',
    'Part proprio en rés. principales 2022',
    'Part locataires HLM dans les rés. principales 2022',
    'Part cadres sup. 2022', 'Part logements vacants 2022',
    'Taux de chômage annuel moyen 2024',
    'Part des élèves du privé parmi les élèves du second degré 2024',
    "Nombre d'établissements 2023","share_appartement","share_vente"
]

    ID_COL = 'nom_gare'
    DATE_COL = "date"
    TARGET_COL ='prix_m2'

    X_list, y_list = [], []

    for _, group in df_split.groupby(ID_COL):
            group = group.sort_values(DATE_COL)

    features = group[num_cols].values      # (T, n_features) avec NaN possibles
    target = group[TARGET_COL].values         # (T,)
    T = len(group)

    for t in range(seq_len, T - horizon + 1):
            x_sequence = features[t-seq_len : t, :]      # shape : (seq_len, n_features)
            y_value = target[t : t+horizon]           # shape : (horizon,)

            # On ne garde la séquence que si toutes les cibles sont observées
            if np.isnan(y_value).any():
                continue


            X_list.append(x_sequence)
            y_list.append(y_value)

    X = np.array(X_list)             # shape : (n_samples, seq_len, n_features)
    y = np.array(y_list)             # shape : (n_samples, horizon)
    return X, y
