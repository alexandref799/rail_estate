import pandas as pd
import glob
import unicodedata

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by 
    - Clean and convert numerical columns 
    - Filter residential properties only 
    - Remove anomalies
    - Keep only useful columns 
    """
    
    # Clean and convert numerical columns 
    
    ## Remove spaces and convert to numeric
    df["Valeur fonciere"] = (
        df["Valeur fonciere"]
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    df["Surface reelle bati"] = (
        df["Surface reelle bati"]
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .astype(float)
    )

    ## Convert date
    df["Date mutation"] = pd.to_datetime(df["Date mutation"], errors="coerce")

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
    "annee",
    "Valeur fonciere",
    "Surface reelle bati",
    "prix_m2",
    "Nombre pieces principales",
    "Type local",
    "No voie",
    "Type de voie",
    "Code voie",
    "Voie",
    "Code postal",
    "Commune",
    "Code departement",
    "Code commune"
    ]
    
    df_clean = df[cols_to_keep].copy()
    
    return df_clean
