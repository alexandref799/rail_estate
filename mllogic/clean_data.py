
import pandas as pd

def clean_data_ban(df: pd.DataFrame) -> pd.DataFrame:

    cols_to_keep = ['numero','nom_voie','code_postal','nom_commune','lon','lat' ]
    df_ban_clean = df[cols_to_keep]

    return df_ban_clean
    
def clean_data_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by
    - Clean and convert numerical columns
    - Filter residential properties only
    - Remove anomalies
    - Keep only useful columns
    """

    cols = [
        'Date mutation',
        'Nature mutation',
        'Type local',
        'Nombre pieces principales',
        'Surface reelle bati',
        'Valeur fonciere',
        'No voie',
        'Type de voie',
        'Voie',
        'Code postal',
        'Commune',
        'Code departement',
        'Code commune'
    ]

    dtypes = {
        'Nature mutation': "string",
        'Type local': "string",
        'Nombre pieces principales': "float64",
        'Surface reelle bati': "string",
        'Valeur fonciere': "string",
        'No voie': "string",
        'Type de voie': "string",
        'Voie': "string",
        'Code postal': "string",
        'Commune': "string",
        'Code departement': "string",
        'Code commune': "string"
    }

    df = df[cols].copy()

    df = df.astype(dtypes)

    # Clean and convert numerical columns

    ## Remove spaces and convert to numeric
    df["Valeur fonciere"] = (
        df["Valeur fonciere"]
            .astype(object) # on force l'homogénéité
            .fillna("")                 # NaN → string vide
            .map(str)                   # conversion sûre en string
            .str.replace(" ", "", regex=False)
            .str.replace(",", ".", regex=False)
    )
    df["Valeur fonciere"] = pd.to_numeric(df["Valeur fonciere"], errors="coerce")

    df["Surface reelle bati"] = (
        df["Surface reelle bati"]
            .astype(object)             # on force l'homogénéité
            .fillna("")                 # NaN → string vide
            .map(str)                   # conversion sûre en string
            .str.replace(" ", "", regex=False)
            .str.replace(",", ".", regex=False)
    )
    df["Surface reelle bati"] = pd.to_numeric(df["Surface reelle bati"], errors="coerce")

    ## Convert date
    df["Date mutation"] = pd.to_datetime(
        df["Date mutation"],
        format="%d/%m/%Y",
        errors="coerce"
    )

    ## Extract columns
    df["annee"] = df["Date mutation"].dt.year
    df['mois'] = df["Date mutation"].dt.month

    # Filter residential properties only
    allowed_local_types = ["Appartement", "Maison"]
    df = df[df["Type local"].isin(allowed_local_types)]

    ####### Filter ventes classiques
    allowed_mutation_types = ["Vente", "Vente en l'état futur d'achèvement"]
    df = df[df["Nature mutation"].isin(allowed_mutation_types)]

    #Remove anomalies
    df = df[df["Valeur fonciere"] > 1000]  # eliminate invalid values
    df = df[df["Surface reelle bati"] > 8]  # eliminate caves / erreurs
    df = df[df["Surface reelle bati"] < 300]  # remove mansions / errors
    df = df[df['Nombre pieces principales'] < 10]  # remove human errors

    ## Create price/m2
    df["prix_m2"] = df["Valeur fonciere"] / df["Surface reelle bati"]

    ## Remove abnormal price/m²
    df = df[df["prix_m2"] > 1000]       # avoid garages
    df = df[df["prix_m2"] < 18000]     # avoid aberrations

    # --- Harmonisation "Commune" en lowercase ---
    df["Commune"] = df["Commune"].astype(str).str.lower().str.strip()

    # --- Correction des codes commune pour Paris ---
    mask_paris = df["Commune"].str.contains("paris ")
    df.loc[mask_paris, "Code commune"] = "056"   # règle DVF : Paris = 056

    # --- Conversion en string + padding ---
    df["Code departement"] = df["Code departement"].astype(str).str.zfill(2)
    df["Code commune"] = df["Code commune"].astype(str).str.zfill(3)

    # --- Création du Code ville ---
    df["Code ville"] = df["Code departement"] + df["Code commune"]

    # --- On affiche la nouvelle colonne ---
    print(df["Code ville"].head())

    cols_to_keep = [
    "Date mutation",
    "annee",
    "mois",
    "Nature mutation",
    "Type local",
    "Valeur fonciere",
    "Surface reelle bati",
    "prix_m2",
    "Nombre pieces principales",
    "No voie",
    "Type de voie",
    "Voie",
    "Code postal",
    "Commune",
    "Code departement",
    "Code commune",
    "Code ville"
    ]

    df_clean = df[cols_to_keep].copy()

    return df_clean

def clean_data_gares(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by
    - Clean geo point -> lat / lon
    - Normalize and rename columns
    - Remove anomalies
    - Keep only useful columns
    """

    # Clean geo point and transform in lat / lon

    def extract_lat_lon(geo):
        try:
            lat, lon = geo.split(",")
            return float(lat), float(lon)
        except:
            return None, None

    df["latitude"], df["longitude"] = zip(*df["Geo Point"].apply(extract_lat_lon))

    # Normalize & rename columns

    df_gares = df.rename(columns={
    "LIBELLE": "nom_gare",
    "CONNEX": "interconnexion",
    "LIGNE": "lignes",
    "Date de signature": "date_signature",
    "Date ouverture officielle": "date_ouverture"
    })

    df_gares.columns = df_gares.columns.str.lower()  # tout en minuscule

    # Clean interconexion
    df_gares["interconnexion"] = (
    df_gares["interconnexion"]
    .str.strip()
    .str.lower()
    .map({"interconnexion": "oui", "non": "non"})
    )

    # Parse dates
    for col in ["date_signature", "date_ouverture"]:
        df_gares[col] = pd.to_datetime(df_gares[col], errors="coerce")

    # Clean spaces
    df_gares["ligne_clean"] = df_gares["lignes"].str.replace(" ", "", regex=False)

    # Selection des colonnes

    colonnes_finales = [
        "nom_gare",
        "latitude",
        "longitude",
        "interconnexion",
        "ligne_clean",
        "date_signature",
        "date_ouverture",
    ]

    df_gares_clean = df_gares[colonnes_finales].copy()

    return df_gares_clean
