import pandas as pd
from typing import List

def assemble_dataframes(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    date_cols: List[str] = ['date_signature', 'date_ouverture']
) -> pd.DataFrame:
    """
    Assemble deux DataFrames et garantit que les colonnes de date spécifiées
    sont au format datetime64[ns] avec l'heure à 00:00:00 (YYYY-MM-DD 00:00:00).

    Args:
        df1 (pd.DataFrame): Le premier DataFrame.
        df2 (pd.DataFrame): Le second DataFrame, dont les colonnes de date sont sécurisées.
        date_cols (List[str]): Liste des noms de colonnes de date à traiter.

    Returns:
        pd.DataFrame: Le nouveau DataFrame combiné et uniformisé.
    """

    # 1. SÉCURISATION ET NORMALISATION DES DATES DANS df2 AVANT CONCAT
    print("Sécurisation du format de date dans df2 avant concaténation...")
    df2_secured = df2.copy()

    for col in date_cols:
        if col in df2_secured.columns:
            # Conversion explicite en datetime64[ns]
            df2_secured[col] = pd.to_datetime(df2_secured[col], errors='coerce')

            # NORMALISATION : Garantir l'heure à 00:00:00
            if df2_secured[col].dt.tz is not None:
                # Retirer le fuseau horaire avant la normalisation
                df2_secured[col] = df2_secured[col].dt.tz_localize(None)

            df2_secured[col] = df2_secured[col].dt.normalize() # <-- GARANTIT 00:00:00
            print(f"   -> Colonne '{col}' de df2 convertie et normalisée.")
        else:
            print(f"   -> Colonne de date '{col}' non trouvée dans df2. Ignorée.")

    # 2. Concaténation
    df_combine = pd.concat([df1, df2_secured], ignore_index=True)

    # 3. TRAITEMENT POST-CONCAT : Uniformisation finale de l'ensemble des données
    # Cette étape est redondante mais agit comme sécurité finale.
    print("\nUniformisation finale des Time Zones sur le DataFrame combiné...")

    for col in date_cols:
        if col in df_combine.columns and df_combine[col].dtype == 'datetime64[ns]':
            try:
                # Retirer le fuseau horaire si présent (sécurité)
                if df_combine[col].dt.tz is not None:
                    df_combine[col] = df_combine[col].dt.tz_localize(None)
                    print(f"   -> Fuseau horaire retiré pour '{col}'.")

                # NORMALISATION FINALE : Garantir l'heure à 00:00:00
                df_combine[col] = df_combine[col].dt.normalize() # <-- Re-garantit 00:00:00

                print(f"   -> Colonne '{col}' uniformisée en type : {df_combine[col].dtype}")

            except AttributeError:
                print(f"   -> ❌ ERREUR D'UNIFORMISATION : La colonne '{col}' n'a pas pu être traitée par '.dt'.")

    print(f"\n✅ DataFrames combinés avec succès. Format de date garanti au 00:00:00.")
    print(f"Taille du DataFrame combiné: {len(df_combine):,} lignes.")

    return df_combine
