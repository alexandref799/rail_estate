import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
from math import radians
import unicodedata
import glob

# ------------------------------------------------------------
# 1. Ajout des coordonnées gps aux transactions
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# 2. Trouver la gare la plus proche via BallTree
# ------------------------------------------------------------

def find_nearest_station(df_dvf_gps: pd.DataFrame, df_gares: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la gare la plus proche + distance haversine (km)
    RETURN → df_dvf_gps_gare
    """

    # Stations coords
    gares_coords = np.radians(df_gares[["latitude", "longitude"]].values)

    tree = BallTree(gares_coords, metric="haversine")

    # DVF coords
    dvf_coords = np.radians(df_dvf_gps[["lat", "lon"]].values)

    dist, idx = tree.query(dvf_coords, k=1)
    dist_km = dist.flatten() * 6371  # Earth radius km

    df_dvf_gps["distance_gare_km"] = dist_km
    df_dvf_gps["nearest_gare"] = df_gares.iloc[idx.flatten()]["nom_gare"].values

    # Merge infos gares
    df_dvf_gps_gare = df_dvf_gps.merge(
        df_gares[["nom_gare", "date_signature", "date_ouverture"]],
        left_on="nearest_gare",
        right_on="nom_gare",
        how="left"
    ).drop(columns=["nom_gare"])

    return df_dvf_gps_gare

# ------------------------------------------------------------
# 3. Relative years (signature & ouverture)
# ------------------------------------------------------------

def compute_relative_years(df_dvf_gps_gare: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute deux colonnes :
    - relative_signature : année DVF - année signature
    - relative_opening   : année DVF - année d’ouverture
    """

    df_dvf_gps_gare["annee"] = pd.to_datetime(df_dvf_gps_gare["Date mutation"]).dt.year
    df_dvf_gps_gare["year_signature"] = pd.to_datetime(df_dvf_gps_gare["date_signature"]).dt.year
    df_dvf_gps_gare["year_opening"] = pd.to_datetime(df_dvf_gps_gare["date_ouverture"]).dt.year

    df_dvf_gps_gare["relative_signature"] = df_dvf_gps_gare["annee"] - df_dvf_gps_gare["year_signature"]
    df_dvf_gps_gare["relative_opening"] = df_dvf_gps_gare["annee"] - df_dvf_gps_gare["year_opening"]

    return df_dvf_gps_gare

# ------------------------------------------------------------
# 4. Drop multicolinéarité
# ------------------------------------------------------------

def drop_multicollinearity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les colonnes redondantes :
    ['Date mutation', 'Valeur fonciere']
    """

    cols_to_drop = ["Date mutation", "Valeur fonciere", 'nearest_gare','date_signature','date_ouverture','year_signature','year_opening']
    df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    return df_clean

# ------------------------------------------------------------
# 5. Pipeline complet (à appeler depuis train.py)
# ------------------------------------------------------------

def run_feature_engineering(df_dvf, df_gares, df_ban):
    """
    Enchaîne :
    1. Ajout GPS
    2. Gare la plus proche
    3. Relative years
    4. Nettoyage final

    RETURN → df_final prêt pour modèle
    """

    print("📍 Ajout des coordonnées GPS...")
    df1 = add_gps_coordinates(df_dvf, df_ban)

    print("🚇 Calcul gare la plus proche...")
    df2 = find_nearest_station(df1, df_gares)

    print("⏳ Ajout des relative years...")
    df3 = compute_relative_years(df2)

    print("🧹 Drop multicolinéarité...")
    df4 = drop_multicollinearity(df3)

    print("✅ Feature engineering terminé.")
    return df4
