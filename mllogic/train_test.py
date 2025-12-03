
import pandas as pd
# Plus besoin de sklearn.model_selection.train_test_split

def train_test_split_strict_chrono(
    df: pd.DataFrame,
    date_col: str,
    min_year: int,
    max_year: int,
    test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    1. Filtre un DataFrame pour inclure les années entre MIN_YEAR et MAX_YEAR.
    2. Applique un split STRICTEMENT CHRONOLOGIQUE (temps passé pour train, temps futur pour test).

    Args:
        df: Le DataFrame (la colonne date_col doit être de type datetime).
        date_col: Nom de la colonne de date.
        min_year: Année de début du filtre (incluse).
        max_year: Année de fin du filtre (incluse).
        test_size: Proportion des données pour le jeu de test (le plus récent).

    Returns:
        X_train, X_test, y_train, y_test
    """

    # 1. Filtrage Chronologique par Année
    df_filtered = df[
        (df[date_col] >= min_year) &
        (df[date_col] <= max_year)
    ].copy()

    print(f"✅ Données filtrées de {min_year} à {max_year}. Total: {len(df_filtered):,} lignes.")

    if df_filtered.empty:
        print("⚠️ Le DataFrame filtré est vide.")
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype='object'), pd.Series(dtype='object')

    # 2. Tri strict par date (NÉCESSAIRE pour le split chronologique)
    df_sorted = df_filtered.sort_values(by=date_col).reset_index(drop=True)
    # 3. Calculer l'index de séparation
    # split_index = 80% des données (le passé)
    split_index = int(len(df_sorted) * (1 - test_size))

    # 4. Découpage Chronologique du DataFrame trié
    train_df = df_sorted.iloc[:split_index] # 0% jusqu'à split_index (Passé)
    test_df = df_sorted.iloc[split_index:]  # split_index jusqu'à 100% (Futur)

    # 5. Séparation en X (Features) et y (Cible)
    y_train = train_df["prix_m2"]
    X_train = train_df.drop(columns=["prix_m2"])

    y_test = test_df["prix_m2"]
    X_test = test_df.drop(columns=["prix_m2"])

    print(f"🎉 Split Chronologique réussi. Train: {len(X_train):,} / Test: {len(X_test):,}")

    return X_train, X_test, y_train, y_test
