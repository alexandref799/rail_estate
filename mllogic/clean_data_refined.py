import pandas as pd

def filter_residential_properties_only(df):
    allowed_local_types = ["Appartement", "Maison"]
    df = df[df["Type local"].isin(allowed_local_types)]
    return df

def drop_too_much_na(df):
    df = df.drop(columns=['Code service CH', 'Reference document', '1 Articles CGI', '2 Articles CGI', '3 Articles CGI', '4 Articles CGI', '5 Articles CGI', 'B/T/Q', 'Prefixe de section', 'No Volume', 'Identifiant local'])
    return df

def drop_lignes_with_na(df):
    df = df.dropna(subset=['Valeur fonciere', 'Surface reelle bati'])
    return df