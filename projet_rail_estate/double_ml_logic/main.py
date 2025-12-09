# mllogic/double_ml/main.py

"""
Script principal pour lancer un pipeline Double ML complet sur le projet Rail Estate.

1. Charge le panel (transactions agrégées ou panel station-année).
2. Prépare le DataFrame DML (y, d, X).
3. Construit les matrices X, y, d (encodage).
4. Lance le Double ML.
5. Sauvegarde les résultats pour Streamlit.
6. Affiche un plot rapide de l'ATE.

À adapter :
- PATH_PANEL
- OUTCOME_COL
- TREATMENT_COL
- FEATURE_COLS
- CATEGORICAL_COLS
"""

from pathlib import Path

import pandas as pd

from . import (
    feature_engineering,
    load_data,
    model,
    preprocessing,
    save,
    visualisation,
)

# ---- PARAMETRES A ADAPTER A TON PANEL ----

PATH_PANEL = Path("data/panel.parquet")

# Exemple cohérent avec ton projet (à adapter à ton vrai panel)
OUTCOME_COL = "prix_m2_mean"        # y
TREATMENT_COL = "treated_gp_1km"    # d (1 = dans zone GP, 0 = contrôle)
FEATURE_COLS = [
    "annee",
    "distance_gare_km",
    "relative_signature_year",
    "Revenu médian 2021",
    "Taux de chômage annuel moyen 2024",
    "departement",
    "ligne_clean",
]
CATEGORICAL_COLS = [
    "departement",
    "ligne_clean",
]

def run_pipeline(version: str = "dev") -> dict:
    # 1. Load panel
    print(f"Loading panel from {PATH_PANEL}...")
    panel = load_data.load_panel(PATH_PANEL)

    # 2. Préparation du df DML
    print("Preparing DML dataframe...")
    dml_df, features_effective = preprocessing.prepare_dml_dataframe(
        panel,
        outcome_col=OUTCOME_COL,
        treatment_col=TREATMENT_COL,
        feature_cols=FEATURE_COLS,
        dropna=True,
    )

    # Ajuster la liste des caté en fonction des features encore présentes
    categorical_effective = [c for c in CATEGORICAL_COLS if c in features_effective]

    # 3. Construction X, y, d
    print("Building design matrices...")
    X, y, d, preprocessor, feature_names_out = feature_engineering.build_design_matrices(
        dml_df,
        outcome_col=OUTCOME_COL,
        treatment_col=TREATMENT_COL,
        feature_cols=features_effective,
        categorical_cols=categorical_effective,
    )

    # 4. Double ML
    print("Running Double ML...")
    dml_res = model.run_double_ml(X, y, d, n_splits=5, random_state=42)

    print("Double ML result:")
    print(dml_res)

    # 5. Sauvegarde résultats (pour Streamlit)
    res_dict = model.dml_result_to_dict(dml_res)
    save.save_results(res_dict, name="dml_global", version=version)

    # 6. Petit plot
    visualisation.plot_ate_point(
        ate=dml_res.ate,
        ci_low=dml_res.ci_low,
        ci_high=dml_res.ci_high,
        title=f"ATE global (Double ML) – version {version}",
    )

    return res_dict


if __name__ == "__main__":
    run_pipeline(version="v0")
