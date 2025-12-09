import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from typing import List, Optional

def run_dml_sklearn_linear(
    df: pd.DataFrame,
    y_col: str,
    d_col: str,
    x_cols: list,
    n_splits: int = 2,
    ml_y=None,
    ml_d=None,
    random_state: int = 42,
) -> dict:
    """
    Double Machine Learning (DML) simple avec scikit-learn.

    Paramètres
    ----------
    df : DataFrame
        Données sources.
    y_col : str
        Nom de la colonne outcome Y (ex. 'log_prix_m2').
    d_col : str
        Nom de la colonne traitement D (ex. 'D_gp_1km' binaire).
    x_cols : list of str
        Liste des colonnes X (features de contrôle).
    n_splits : int
        Nombre de folds pour le cross-fitting.
    ml_y : sklearn estimator or None
        Modèle pour E[Y|X]. Si None -> RandomForestRegressor par défaut.
    ml_d : sklearn estimator or None
        Modèle pour E[D|X]. Si None -> RandomForestRegressor par défaut.
    random_state : int
        Graine pour KFold et les modèles par défaut.

    Retourne
    --------
    dict
        Résultats avec tau_hat, se, IC, etc.
    """

    # 1. Préparation des données
    cols = [y_col, d_col] + x_cols
    data = df[cols].dropna().copy()

    y = data[y_col].astype(float).to_numpy()
    d = data[d_col].astype(float).to_numpy()
    X = data[x_cols].to_numpy()
    n = X.shape[0]

    if ml_y is None:
        ml_y = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        )

    if ml_d is None:
        ml_d = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            random_state=random_state + 1,
            n_jobs=-1,
        )

    # 2. Cross-fitting pour obtenir m_hat(X) et g_hat(X) sur tout l'échantillon
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    m_hat = np.zeros(n)
    g_hat = np.zeros(n)

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, d_train = y[train_idx], d[train_idx]

        # m(X) = E[Y|X]
        model_y = clone(ml_y)
        model_y.fit(X_train, y_train)
        m_hat[test_idx] = model_y.predict(X_test)

        # g(X) = E[D|X]
        model_d = clone(ml_d)
        model_d.fit(X_train, d_train)
        g_hat[test_idx] = model_d.predict(X_test)

    # 3. Résidualisation
    y_tilde = y - m_hat
    d_tilde = d - g_hat

    # 4. Régression linéaire de y_tilde sur d_tilde
    #    (sans intercept, car intercept implicite dans la construction DML)
    d_tilde_2d = d_tilde.reshape(-1, 1)
    ols = LinearRegression(fit_intercept=False)
    ols.fit(d_tilde_2d, y_tilde)

    tau_hat = float(ols.coef_[0])

    # 5. Écart-type robuste (formule OLS simple, 1 régresseur)
    y_hat = ols.predict(d_tilde_2d)
    resid = y_tilde - y_hat
    # variance des résidus
    sigma2 = float(np.sum(resid ** 2) / (n - 1))
    # variance de tau_hat
    var_tau = sigma2 / float(np.sum(d_tilde ** 2))
    se_tau = float(np.sqrt(var_tau))

    # Intervalle de confiance 95% (approx. normale)
    z = 1.96
    ci_low = tau_hat - z * se_tau
    ci_high = tau_hat + z * se_tau

    results = {
        "tau_hat": tau_hat,
        "se_tau": se_tau,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": int(n),
        "n_splits": n_splits,
        "y_col": y_col,
        "d_col": d_col,
        "x_cols": list(x_cols),
        "random_state": random_state,
    }

    return results


def run_dml_by_station(
    df: pd.DataFrame,
    y_col: str,
    base_treat_col: str,
    station_col: str,
    x_cols: List[str],
    n_splits: int = 2,
    ml_y=None,
    ml_d=None,
    random_state: int = 42,
    min_treated: int = 100,
) -> pd.DataFrame:
    """
    Boucle DML par gare.

    Pour chaque gare, on définit un traitement D_station = 1 si la transaction
    est dans le buffer 1 km de cette gare, 0 sinon (reste de l'IDF),
    puis on appelle run_dml_sklearn_linear.

    Paramètres
    ----------
    df : DataFrame
        Panel complet avec au moins y_col, base_treat_col, station_col et X.
    y_col : str
        Nom de la cible (ex : 'log_prix_m2').
    base_treat_col : str
        Colonne binaire globale (ex : 'is_gp_1km' ou 'D_gp_1km') qui vaut 1
        si la transaction est à ≤ 1 km d'une nouvelle gare.
    station_col : str
        Colonne qui identifie la gare la plus proche (ex : 'station_id').
    x_cols : list of str
        Liste des features X.
    n_splits, ml_y, ml_d, random_state :
        Paramètres passés à run_dml_sklearn_linear.
    min_treated : int
        Nombre minimum d'observations traitées (D=1) pour garder la gare.

    Retour
    ------
    DataFrame
        Une ligne par gare avec tau_hat, IC, n, + ATE en %.
    """

    results = []
    stations = df[station_col].dropna().unique()

    for s in stations:
        df_s = df.copy()

        # D_station = 1 si (dans le buffer GP) ET (gare la plus proche = s)
        df_s["D_station"] = np.where(
            (df_s[base_treat_col] == 1) & (df_s[station_col] == s),
            1.0,
            0.0,
        )

        n_treated = int(df_s["D_station"].sum())
        n_total = int(df_s.shape[0])

        # skip si trop peu de traités ou si D constant
        if n_treated < min_treated or n_treated == 0 or n_treated == n_total:
            continue

        dml_res = run_dml_sklearn_linear(
            df=df_s,
            y_col=y_col,
            d_col="D_station",
            x_cols=x_cols,
            n_splits=n_splits,
            ml_y=ml_y,
            ml_d=ml_d,
            random_state=random_state,
        )

        tau = dml_res["tau_hat"]
        ci_low = dml_res["ci_low"]
        ci_high = dml_res["ci_high"]

        # conversion log -> % de prix/m²
        ate_pct = (np.exp(tau) - 1) * 100
        ci_low_pct = (np.exp(ci_low) - 1) * 100
        ci_high_pct = (np.exp(ci_high) - 1) * 100

        results.append(
            {
                "station_id": s,
                "tau_hat": tau,
                "se_tau": dml_res["se_tau"],
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ate_pct": ate_pct,
                "ci_low_pct": ci_low_pct,
                "ci_high_pct": ci_high_pct,
                "n": dml_res["n"],
                "n_treated": n_treated,
            }
        )

    return pd.DataFrame(results)

