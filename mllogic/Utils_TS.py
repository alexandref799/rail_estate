

def group_by(
    df: pd.DataFrame,
    colonne_date: str,
    colonne_gare: str,
    colonnes_numeriques: list,
    colonnes_categorielles: list,
):
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
        raise ValueError("One of the splits is empty; adjust test_start/test_end.")

    return df_train, df_test


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
