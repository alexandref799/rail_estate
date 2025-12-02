
import pandas as pd
import glob
import unicodedata

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

    ## Extract year
    df["annee"] = df["Date mutation"].dt.year

    # Filter residential properties only
    allowed_types = ["Appartement", "Maison"]
    df = df[df["Type local"].isin(allowed_types)]

    #Remove anomalies
    df = df[df["Valeur fonciere"] > 1000]  # eliminate invalid values
    df = df[df["Surface reelle bati"] > 8]  # eliminate caves / erreurs
    df = df[df["Surface reelle bati"] < 500]  # remove mansions / errors

    ## Create price/m2
    df["prix_m2"] = df["Valeur fonciere"] / df["Surface reelle bati"]

    ## Remove abnormal price/m²
    df = df[df["prix_m2"] > 500]       # avoid garages
    df = df[df["prix_m2"] < 20000]     # avoid aberrations

    cols_to_keep = [
    "Date mutation",
    "Nature mutation",
    "annee",
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
    "Code commune"
    ]

    df_clean = df[cols_to_keep].copy()

    return df_clean
