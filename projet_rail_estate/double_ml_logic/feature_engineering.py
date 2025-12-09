# mllogic/double_ml/feature_engineering.py

from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import requests
import numpy as np
import pandas as pd
import time
import unicodedata

from sklearn.neighbors import BallTree


def merge_all_gares(df_new_gares_clean: pd.DataFrame,df_old_gares_clean: pd.DataFrame) -> pd.DataFrame:

    # Colonnes communes
    gp_cols_for_all = ["id_gare", "nom_gare", "latitude", "longitude", "is_grand_paris", "annee_signature"]
    old_cols_for_all = ["id_gare", "nom_gare", "latitude", "longitude", "is_grand_paris"]

    # Pour les anciennes gares, pas de date de signature
    old_for_all = df_old_gares_clean[old_cols_for_all].copy()
    old_for_all["annee_signature"] = np.nan

    gp_for_all = df_new_gares_clean[gp_cols_for_all].copy()

    df_gares_all = pd.concat([gp_for_all, old_for_all], ignore_index=True)

    df_gares_all.head(), df_gares_all["is_grand_paris"].value_counts()

    return df_gares_all

def reverse_new_gares_ban(df_new_gares_clean):
    """
    Appelle l'API BAN pour récupérer ville, code commune, code postal, département
    à partir de (lat, lon).
    """

    lon_col = df_new_gares_clean['longitude']
    lat_col = df_new_gares_clean['latitude']

    def fonction_retrieve_api(lat_col,lon_col):

        url = "https://api-adresse.data.gouv.fr/reverse/"
        params = {"lat": lat_col, "lon": lon_col}

        try:
            r = requests.get(url, params=params, timeout=5)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print("Erreur BAN pour", lat_col, lon_col, ":", e)
            return pd.Series({
                "code_commune": np.nan,
                "ville": np.nan,
                "code_postal": np.nan,
                "departement": np.nan,
            })

        if not data.get("features"):
            return pd.Series({
                "code_commune": np.nan,
                "ville": np.nan,
                "code_postal": np.nan,
                "departement": np.nan,
            })

        props = data["features"][0]["properties"]
        code_commune = props.get("citycode")
        ville        = props.get("city")
        code_postal  = props.get("postcode")

        # pour la France métropolitaine : département = 2 premiers chiffres du CP
        if isinstance(code_postal, str) and len(code_postal) >= 2:
            departement = code_postal[:2]
        else:
            departement = np.nan

        return pd.Series({
            "code_commune": code_commune,
            "ville": ville,
            "code_postal": code_postal,
            "departement": departement,
        })

    df_new_gares_clean[["code_commune", "ville", "code_postal", "departement"]] = df_new_gares_clean.apply(lambda row: fonction_retrieve_api(row[lat_col], row[lon_col]), axis=1)
    df_new_gares_codes = df_new_gares_clean.copy()

    return df_new_gares_codes

def add_log_price(dvf):

    dvf["log_price_m2"] = np.log(dvf["prix_m2"])

    return dvf

def add_gps_coordinates(df_dvf_clean, df_ban_clean):

    # -------------------------------------------------------------
    # 0. Dictionnaire des types de voie
    # -------------------------------------------------------------
    dict_type_voie = {
        'ALL': 'ALLEE', 'AV': 'AVENUE',
        'BD': 'BOULEVARD', 'BER': 'BERGE', 'BORD': 'BORD', 'CAR': 'CARREFOUR',
        'CC': 'CENTRE COMMERCIAL', 'CD': 'CHEMIN DEPARTEMENTAL', 'CHE': 'CHEMIN',
        'CHM': 'CHEMIN', 'CHS': 'CHAUSSEE', 'CHT': 'CHALET', 'CHV': 'CHEMIN VICINAL',
        'CITE': 'CITE', 'CLOS': 'CLOS', 'COTE': 'COTE', 'COUR': 'COUR',
        'CR': 'CHEMIN RURAL', 'CRS': 'COURS', 'CRX': 'CROIX', 'CTR': 'CENTRE',
        'D': 'DOMAINE', 'DOM': 'DOMAINE', 'ESC': 'ESCALIER', 'ESP': 'ESPLANADE',
        'FG': 'FAUBOURG', 'FRM': 'FERME', 'GAL': 'GALERIE', 'GPL': "GROUPE D IMMEUBLES",
        'GR': 'GRANDE RUE', 'HAM': 'HAMEAU', 'HLM': 'HABITATION A LOYER MODERE',
        'IMP': 'IMPASSE', 'JARD': 'JARDIN', 'LOT': 'LOTISSEMENT',
        'MAIL': 'MAIL', 'MTE': 'MONTEE', 'PKG': 'PARKING', 'PL': 'PLACE',
        'PLA': 'PLATEAU', 'PLE': 'PETITE LEVEE', 'PONT': 'PONT',
        'PORT': 'PORT', 'PROM': 'PROMENADE', 'PRV': 'PARVIS',
        'PTTE': 'PETITE ROUTE', 'QUA': 'QUARTIER', 'QUAI': 'QUAI',
        'RES': 'RESIDENCE', 'RLE': 'RUELLE', 'ROC': 'ROCADE',
        'RTE': 'ROUTE', 'RUE': 'RUE', 'SEN': 'SENTE', 'SQ': 'SQUARE',
        'TRA': 'TRAVERSE', 'TRS': 'TERRASSE', 'VAL': 'VAL', 'VALL': 'VALLEE',
        'VGE': 'VILLAGE', 'VIL': 'VILLE', 'VLA': 'VILLA', 'VOIE': 'VOIE',
        'VOIR': 'VOIRIE', 'VTE': 'VENTE', 'ZA': 'ZA', 'ZAC': 'ZAC'
    }

    # -------------------------------------------------------------
    # 1. Fonction de normalisation
    # -------------------------------------------------------------
    def normalize(s):
        if pd.isna(s):
            return ""
        s = s.upper().strip()
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")
        return s

    # -------------------------------------------------------------
    # 2. Reconstruction d’adresse DVF
    # -------------------------------------------------------------

    df_dvf_clean["type_voie_expanded"] = (
        df_dvf_clean["Type de voie"].map(dict_type_voie).fillna(df_dvf_clean["Type de voie"])
    )

    df_dvf_clean["numero"] = (
        df_dvf_clean["No voie"].fillna("").astype(str).str.replace(r"\.0$", "", regex=True)
    )

    df_dvf_clean["nom_voie"] = (
        df_dvf_clean["type_voie_expanded"].fillna("") + " " + df_dvf_clean["Voie"].fillna("")).apply(normalize)

    df_dvf_clean["code_postal"] = (
        df_dvf_clean["Code postal"].astype(str).str.replace(r"\.0$", "", regex=True).str[:5]
    )

    df_dvf_clean["adresse_complete"] = df_dvf_clean["numero"].astype(str) + " " + df_dvf_clean["nom_voie"].astype(str) + " " + df_dvf_clean["code_postal"].astype(str)
    df_dvf_clean["adresse_complete"] = df_dvf_clean["adresse_complete"].str.upper()

    df_ban_clean["adresse_complete"] = df_ban_clean["numero"].astype(str) + " " + df_ban_clean["nom_voie"].astype(str) + " " + df_ban_clean["code_postal"].astype(str)
    df_ban_clean["adresse_complete"] = df_ban_clean["adresse_complete"].str.upper()

    # -------------------------------------------------------------
    # 3. Merge avec BAN
    # -------------------------------------------------------------
    df_dvf_gps = df_dvf_clean.merge(df_ban_clean, on="adresse_complete", how="inner")

    df_dvf_gps = df_dvf_gps.drop(columns=['No voie', 'Type de voie', 'Voie', 'Code postal', 'Commune',
       'Code departement', 'Code commune', 'type_voie_expanded', 'numero_x',
       'nom_voie_x', 'code_postal_x', 'adresse_complete', 'numero_y',
       'nom_voie_y', 'code_postal_y', 'nom_commune'])

    print("GPS coverage :", 1 - df_dvf_gps[["lon", "lat"]].isna().mean())

    return df_dvf_gps

def prepare_gp(df_gp: pd.DataFrame, df_insee_prep: pd.DataFrame) -> pd.DataFrame:
    """Prépare les 78 gares GP + enrichit avec Code ville et département INSEE."""
    g = df_gp.copy().reset_index(drop=True)
    g["station_id"] = g.index.astype(int)

    g["date_signature"] = pd.to_datetime(g["date_signature"])
    g["date_ouverture"] = pd.to_datetime(g["date_ouverture"])

    g["ville_norm"] = (
        g["ville"]
        .str.upper()
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("ascii")
        .str.strip()
    )

    city = df_insee_prep[["Code ville", "Ville_norm", "Département"]].drop_duplicates("Code ville")
    g = g.merge(city, left_on="ville_norm", right_on="Ville_norm", how="left")

    if "departement" in g.columns:
        g["departement"] = g["departement"].fillna(g["Département"])
    else:
        g["departement"] = g["Département"]

    g["departement"] = g["departement"].astype(str)

    keep_cols = [
        "station_id", "nom_gare", "latitude", "longitude",
        "ligne_clean", "departement",
        "date_signature", "date_ouverture",
        "Code ville"
    ]
    return g[keep_cols]

def prepare_gares_all(df_gares_all: pd.DataFrame) -> pd.DataFrame:
    ga = df_gares_all.copy()
    if "id_gare" not in ga.columns:
        ga = ga.reset_index(drop=True)
        ga["id_gare"] = ga.index.astype(int)
    return ga[["id_gare", "nom_gare", "latitude", "longitude"]]

def attach_nearest_gare(
    df_points: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    df_gares: pd.DataFrame,
    gares_lat_col: str,
    gares_lon_col: str,
    id_col: str,
    prefix: str,
) -> pd.DataFrame:
    df = df_points.copy()

    pts = np.radians(df[[lat_col, lon_col]].to_numpy())
    gares = np.radians(df_gares[[gares_lat_col, gares_lon_col]].to_numpy())

    tree = BallTree(gares, metric="haversine")
    dist_rad, idx = tree.query(pts, k=1)

    df[f"{prefix}_distance_km"] = dist_rad.flatten() * 6371.0
    df[f"{prefix}_id"] = df_gares.iloc[idx.flatten()][id_col].values
    return df

def merge_insee_on_dvf(dvf: pd.DataFrame, df_insee_prep: pd.DataFrame) -> pd.DataFrame:
    df = dvf.copy()
    cols_insee = [
        "Code ville", "Département", "Ville",
        "Revenu médian 2021",
        "Évol. annuelle moy. de la population 2016-2022",
        "Part proprio en rés. principales 2022",
        "Part locataires HLM dans les rés. principales 2022",
        "Part cadres sup. 2022",
        "Part logements vacants 2022",
        "Taux de chômage annuel moyen 2024",
        "Part des élèves du privé parmi les élèves du second degré 2024",
        "Nombre d'établissements 2023",
    ]
    return df.merge(df_insee_prep[cols_insee], on="Code ville", how="left")

def flag_1km_proximity(dvf_enriched):
    dvf_enriched["is_gp_1km"] = dvf_enriched["distance_gp_km"] <= 1.0
    dvf_enriched["is_hist_1km"] = dvf_enriched["distance_hist_km"] <= 1.0

    return dvf_enriched

def build_treated_monthly(treated: pd.DataFrame, df_gp_prep: pd.DataFrame) -> pd.DataFrame:
    df = treated.copy()
    df["date"] = df["Date mutation"].dt.to_period("M").dt.to_timestamp()

    agg = (
        df.groupby(["station_id", "date"])
          .agg(
              prix_m2_mean=("prix_m2", "mean"),
              n_transactions=("prix_m2", "size"),
              revenu_median_2021=("Revenu médian 2021", "mean"),
              taux_chomage_2024=("Taux de chômage annuel moyen 2024", "mean"),
              evo_population_2016_2022=("Évol. annuelle moy. de la population 2016-2022", "mean"),
              part_hlm_2022=("Part locataires HLM dans les rés. principales 2022", "mean"),
              part_cadres_2022=("Part cadres sup. 2022", "mean"),
              part_log_vacants_2022=("Part logements vacants 2022", "mean"),
              part_diplomes_2024=("Part des élèves du privé parmi les élèves du second degré 2024", "mean"),
              nb_entreprises_2023=("Nombre d'établissements 2023", "mean")
          )
          .reset_index()
    )

    g = df_gp_prep[[
        "station_id", "nom_gare", "ligne_clean",
        "departement", "date_signature", "date_ouverture"
    ]].drop_duplicates("station_id")

    agg = agg.merge(g, on="station_id", how="left")
    agg["relative_signature_year"] = agg["date"].dt.year - agg["date_signature"].dt.year

    return agg.sort_values(["station_id", "date"]).reset_index(drop=True)

def build_control_series(control_df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = control_df.copy()
    df["date"] = df["Date mutation"].dt.to_period("M").dt.to_timestamp()
    series = (
        df.groupby("date")
          .agg(
              **{f"prix_{name}": ("prix_m2", "mean")},
              **{f"n_{name}": ("prix_m2", "size")},
          )
          .reset_index()
    )
    return series.sort_values("date")



#############################@


def build_design_matrices(
    df: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    feature_cols: Iterable[str],
    categorical_cols: Iterable[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, ColumnTransformer, List[str]]:
    """
    Construit les matrices X, y, d pour le Double ML.

    - One-hot encode les colonnes catégorielles
    - Standardise les features numériques
    - Retourne X (2D), y (1D), d (1D), transformer (pour reuse) et noms de features encodées.

    Args
    ----
    df : DataFrame préparée (sortie de prepare_dml_dataframe)
    outcome_col : nom de la colonne outcome (y)
    treatment_col : nom de la colonne traitement (d)
    feature_cols : colonnes de features brutes
    categorical_cols : sous-ensemble de feature_cols à encoder (les autres sont numériques)

    Returns
    -------
    X : array (n_samples, n_features_encodées)
    y : array (n_samples,)
    d : array (n_samples,)
    transformer : ColumnTransformer fitted
    feature_names_out : liste des noms de features encodées (approx., via get_feature_names_out)
    """
    feature_cols = list(feature_cols)
    categorical_cols = list(categorical_cols)
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    # Définir les transformers
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    X = preprocessor.fit_transform(df[feature_cols])

    y = df[outcome_col].to_numpy(dtype=float)
    d = df[treatment_col].to_numpy(dtype=float)

    # noms des features encodées (optionnel)
    try:
        feature_names_out = preprocessor.get_feature_names_out()
        feature_names_out = list(feature_names_out)
    except Exception:
        feature_names_out = []

    return X, y, d, preprocessor, feature_names_out
