import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from google.cloud import storage
from load_data import _load_single_csv, load_ban, load_dvf_gares, 
from clean_data import clean_data_dvf, clean_data_gares, clean_data_ban, clean_data_old_gares, clean_taux, clean_insee
from feature_engineering import merge_all_gares, reverse_new_gares_ban, add_log_price, add_gps_coordinates, attach_nearest_gare, prepare_gp, prepare_gares_all, merge_insee_on_dvf, flag_1km_proximity, build_treated_monthly, build_control_series

# Load data
df_taux = _load_single_csv(bucket_name="rail-estate-data", uri_file="taux/data_taux.csv")
df_insee = _load_single_csv(bucket_name="rail-estate-data", uri_file="insee/data_insee.csv")
df_ban = load_ban(bucket_name='rail-estate-data', prefix_ban='ban/')
df_dvf, df_new_gares = load_dvf_gare(bucket_name='rail-estate-data', uri_dvf= 'dvf/dvf_idf_2014_2025.csv', uri_gare = 'gares/csv_gares.csv')
df_old_gares = _load_single_csv(bucket_name="rail-estate-data", uri_file="gares/emplacement-des-gares-idf.csv")

# Clean data
df_dvf_clean = clean_data_dvf(df_dvf)
df_ban_clean = clean_data_ban(df_ban)
df_old_gares_clean = clean_data_old_gares(df_old_gares)
df_new_gares_clean = clean_data_gares(df_new_gares)
df_taux_clean = clean_taux(df_taux)
df_insee_clean = clean_insee(df_insee)

# Feature engineering
merge_all_gares = merge_all_gares(df_old_gares_clean, df_new_gares_clean)
df_gp_prep = prepare_gp(df_new_gares_clean, df_insee_clean)
df_gares_all_prep = prepare_gares_all(merge_all_gares)

# Ajout pour les 78 nouvelles gares pour récup ville, code commune, code post , dept a partir de lat lon
df_new_gares_codes = reverse_new_gares_ban(df_new_gares_clean)
### fonction feature_engineering.prepare_gp pas utilisée ici 

# Ajout des coord GPS sur les dvf
df_dvf_clean = add_log_price(df_dvf_clean)
df_dvf_gps = add_gps_coordinates(df_dvf_clean, df_ban_clean)

# Distances aux gares
## Distances aux gares GP
df_dvf_with_closest_new_gares = attach_nearest_gare(df_dvf_gps, lat_col="lat", lon_col="lon",
    df_gares=df_gp_prep, gares_lat_col="latitude", gares_lon_col="longitude",
    id_col="station_id", prefix="gp")

df_dvf_with_closest_new_gares = df_dvf_with_closest_new_gares.rename(columns={"gp_id": "station_id", "gp_distance_km": "distance_gp_km"})

## Distances aux gares historiques (gares all – gp)
gares_hist = df_gares_all_prep[~df_gares_all_prep["nom_gare"].isin(df_gp_prep["nom_gare"])].copy()

df_dvf_with_closest_all_gares = attach_nearest_gare(
    df_dvf_with_closest_new_gares, lat_col="lat", lon_col="lon",
    df_gares=gares_hist, gares_lat_col="latitude", gares_lon_col="longitude",
    id_col="id_gare", prefix="hist"
)
df_dvf_with_closest_all_gares = df_dvf_with_closest_all_gares.rename(columns={"hist_distance_km": "distance_hist_km"})
dvf_enriched = merge_insee_on_dvf(df_dvf_with_closest_all_gares, df_insee_clean)
dvf_enriched = flag_1km_proximity(dvf_enriched)


dml_df = dvf_enriched.copy()
dml_df["date"] = dml_df["Date mutation"].dt.to_period("M").dt.to_timestamp()
dml_df = dml_df.merge(df_taux_clean, on="date", how="left")
dml_df["D_gp_1km"] = (dml_df["distance_gp_km"] <= 1.0).astype(int)


# Model
dml_result = run_dml_sklearn_linear(
    df=dml_df,
    y_col="log_prix_m2",
    d_col="D_gp_1km",
    x_cols=X_cols,
    n_splits=2,
    random_state=42,
)



# Creation des subsets
treated = dvf_enriched[dvf_enriched["is_gp_1km"]].copy()
control_hist = dvf_enriched[~dvf_enriched["is_gp_1km"] & dvf_enriched["is_hist_1km"]].copy()
control_rest = dvf_enriched[~dvf_enriched["is_gp_1km"]].copy()  # tout ce qui n'est pas à ≤ 1 km des GP

# Panel mensuels autour des nouvelles gares et contrôles globaux
treated_monthly = build_treated_monthly(treated, df_gp_prep)
control_hist_monthly = build_control_series(control_hist, "control_hist")
control_rest_monthly = build_control_series(control_rest, "control_rest")

# Merges globaux
treated_monthly = treated_monthly.merge(df_taux_clean, on="date", how="left")
panel = treated_monthly.merge(control_hist_monthly, on="date", how="left")
panel = panel.merge(control_rest_monthly, on="date", how="left")


# =======
# # mllogic/double_ml/main.py

# """
# Script principal pour lancer un pipeline Double ML complet sur le projet Rail Estate.

# 1. Charge le panel (transactions agrégées ou panel station-année).
# 2. Prépare le DataFrame DML (y, d, X).
# 3. Construit les matrices X, y, d (encodage).
# 4. Lance le Double ML.
# 5. Sauvegarde les résultats pour Streamlit.
# 6. Affiche un plot rapide de l'ATE.

# À adapter :
# - PATH_PANEL
# - OUTCOME_COL
# - TREATMENT_COL
# - FEATURE_COLS
# - CATEGORICAL_COLS
# """

# from pathlib import Path

# import pandas as pd

# from . import (
#     feature_engineering,
#     load_data,
#     model,
#     preprocessing,
#     save,
#     visualisation,
# )

# ---- PARAMETRES A ADAPTER A TON PANEL ----

# PATH_PANEL = Path("data/panel.parquet")

# # Exemple cohérent avec ton projet (à adapter à ton vrai panel)
# OUTCOME_COL = "prix_m2_mean"        # y
# TREATMENT_COL = "treated_gp_1km"    # d (1 = dans zone GP, 0 = contrôle)
# FEATURE_COLS = [
#     "annee",
#     "distance_gare_km",
#     "relative_signature_year",
#     "Revenu médian 2021",
#     "Taux de chômage annuel moyen 2024",
#     "departement",
#     "ligne_clean",
# ]
# CATEGORICAL_COLS = [
#     "departement",
#     "ligne_clean",
# ]

# def run_pipeline(version: str = "dev") -> dict:
#     # 1. Load panel
#     print(f"Loading panel from {PATH_PANEL}...")
#     panel = load_data.load_panel(PATH_PANEL)

#     # 2. Préparation du df DML
#     print("Preparing DML dataframe...")
#     dml_df, features_effective = preprocessing.prepare_dml_dataframe(
#         panel,
#         outcome_col=OUTCOME_COL,
#         treatment_col=TREATMENT_COL,
#         feature_cols=FEATURE_COLS,
#         dropna=True,
#     )

#     # Ajuster la liste des caté en fonction des features encore présentes
#     categorical_effective = [c for c in CATEGORICAL_COLS if c in features_effective]

#     # 3. Construction X, y, d
#     print("Building design matrices...")
#     X, y, d, preprocessor, feature_names_out = feature_engineering.build_design_matrices(
#         dml_df,
#         outcome_col=OUTCOME_COL,
#         treatment_col=TREATMENT_COL,
#         feature_cols=features_effective,
#         categorical_cols=categorical_effective,
#     )

#     # 4. Double ML
#     print("Running Double ML...")
#     dml_res = model.run_double_ml(X, y, d, n_splits=5, random_state=42)

#     print("Double ML result:")
#     print(dml_res)

#     # 5. Sauvegarde résultats (pour Streamlit)
#     res_dict = model.dml_result_to_dict(dml_res)
#     save.save_results(res_dict, name="dml_global", version=version)

#     # 6. Petit plot
#     visualisation.plot_ate_point(
#         ate=dml_res.ate,
#         ci_low=dml_res.ci_low,
#         ci_high=dml_res.ci_high,
#         title=f"ATE global (Double ML) – version {version}",
#     )

#     return res_dict


# if __name__ == "__main__":
#     run_pipeline(version="v0")
# >>>>>>> master
