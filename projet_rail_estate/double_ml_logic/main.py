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


