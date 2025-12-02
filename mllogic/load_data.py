from google.cloud import storage
import pandas as pd
import os

# Initialisation du client GCS une seule fois (meilleure pratique)
storage_client = storage.Client(project='rail-estate')

# --- FONCTION UTILITAIRE (pour lire un fichier unique) ---

def _load_single_csv(bucket_name: str, uri_file: str) -> pd.DataFrame:
    """
    Fonction interne pour charger un fichier CSV unique depuis GCS.
    Elle gère la connexion, la lecture, et les erreurs.
    """
    try:
        bucket = storage_client.bucket(bucket_name)
        print(storage_client.project)
        blob = bucket.blob(uri_file)


        # Ouvre le blob comme un fichier en mémoire et le passe à pandas
        with blob.open("r", encoding="utf-8") as f:
            df = pd.read_csv(f)

        print(f"✅ Fichier chargé : {uri_file} ({len(df):,} lignes)")
        return df

    except Exception as e:
        print(f"❌ ERREUR de chargement pour {uri_file}: {e}")
        # Retourne un DataFrame vide en cas d'échec pour ne pas bloquer le reste
        return pd.DataFrame()

def list_uris_in_bucket(bucket_name: str, prefix: str = '') -> list[str]:
    """Liste tous les chemins d'accès (URIs) des fichiers dans un bucket GCS."""
    try:
        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        uri_list = [blob.name for blob in blobs]
        return uri_list
    except Exception as e:
        print(f"❌ ERREUR lors de la liste des blobs dans le bucket : {e}")
        return []

# -----------------------------------------------------------------
# --- FONCTION PRINCIPALE : CHARGE LES 3 DATAFRAMES EN UNE FOIS ---
# -----------------------------------------------------------------

def load_all_data_final(
    bucket_name: str,
    uri_dvf: str,
    prefix_ban: str,
    uri_gare: str # URI unique pour les gares
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Charge les DataFrames : DVF (simple), BAN (concaténé), Gares (simple)
    depuis GCS.
    """
    print("--- Début du chargement des données (Final) ---")

    # 1. Chargement du DataFrame DVF (simple)
    df_dvf = _load_single_csv(bucket_name, uri_dvf)

    # 2. Chargement et Concaténation du DataFrame BAN (Multiple)

    print("\n--- 2. Chargement Dynamique des DataFrames BAN ---")
    list_uri_ban = list_uris_in_bucket(bucket_name, prefix=prefix_ban)
    # Filtrage pour ne garder que les fichiers CSV dans le dossier
    list_uri_ban = [uri for uri in list_uri_ban if uri != prefix_ban and uri.endswith('.csv')]

    dfs_ban_liste = []
    for uri_ban in list_uri_ban:
        df_temp = _load_single_csv(bucket_name, uri_ban)
        if not df_temp.empty:
            dfs_ban_liste.append(df_temp)

    # Concaténation BAN
    if dfs_ban_liste:
        df_ban_concatene = pd.concat(dfs_ban_liste, ignore_index=True)
        print(f"\n✅ CONCATÉNATION RÉUSSIE : DataFrame BAN ({len(df_ban_concatene):,} lignes au total).")
    else:
        df_ban_concatene = pd.DataFrame()
        print("\n⚠️ Concaténation BAN échouée.")


    # 3. Chargement du DataFrame Gares (Simple - via URI directe du .env)
    print("\n--- 3. Chargement du DataFrame Gares (Fichier Unique) ---")
    df_gare = _load_single_csv(bucket_name, uri_gare)

    print("\n--- Fin du chargement des trois DataFrames ---")

    return df_dvf, df_ban_concatene, df_gare
