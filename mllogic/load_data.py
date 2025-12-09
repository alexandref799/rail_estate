from google.cloud import storage
import pandas as pd
import os

# Initialisation du client GCS une seule fois (meilleure pratique)
storage_client = storage.Client()
# Récupère le projet utilisé par le client (si vous l'avez défini)
print(f"Client initialisé. Le projet par défaut est : {storage_client.project}")

# --- FONCTION UTILITAIRE (pour lire un fichier unique) ---

def _load_single_csv(bucket_name: str, uri_file: str) -> pd.DataFrame:
    """
    Fonction interne pour charger un fichier CSV unique depuis GCS.
    Elle gère la connexion, la lecture, et les erreurs.
    """
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(uri_file)


        # Ouvre le blob comme un fichier en mémoire et le passe à pandas
        with blob.open("rb") as f:
            df = pd.read_csv(f,sep=";")

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
        print(uri_list)
        return uri_list
    except Exception as e:
        print(f"❌ ERREUR lors de la liste des blobs dans le bucket : {e}")
        return []

# -----------------------------------------------------------------
# --- FONCTION PRINCIPALE : CHARGE LES 3 DATAFRAMES EN UNE FOIS ---
# -----------------------------------------------------------------
import pandas as pd

def load_ban(bucket_name: str, prefix_ban: str) -> pd.DataFrame:
    """
    Charge et concatène tous les CSV du dossier `prefix_ban` dans le bucket GCS.
    Retourne un seul DataFrame.
    """
    print("\n--- 2. Chargement dynamique des DataFrames BAN ---")
    list_uri_ban = list_uris_in_bucket(bucket_name, prefix=prefix_ban)

    print("URIs trouvées dans le bucket :")
    for uri in list_uri_ban:
        print("  -", uri)

    # Filtrage pour garder uniquement les fichiers CSV
    list_uri_ban = [
        uri for uri in list_uri_ban
        if uri.endswith(".csv")  # et on se fiche de uri != prefix_ban ici
    ]

    print("\nURIs retenues pour chargement (CSV uniquement) :")
    for uri in list_uri_ban:
        print("  -", uri)

    dfs_ban_liste = []
    for uri_ban in list_uri_ban:
        df_temp = _load_single_csv(bucket_name, uri_ban)
        if df_temp is not None and not df_temp.empty:
            dfs_ban_liste.append(df_temp)
        else:
            print(f"⚠️ Fichier vide ou non chargé : {uri_ban}")

    # Concaténation BAN
    if dfs_ban_liste:
        df_ban_concatene = pd.concat(dfs_ban_liste, ignore_index=True)
        print(f"\n✅ CONCATÉNATION RÉUSSIE : DataFrame BAN ({len(df_ban_concatene):,} lignes au total).")
    else:
        df_ban_concatene = pd.DataFrame()
        print("\n⚠️ Aucun CSV valide trouvé, concaténation BAN impossible (DataFrame vide).")

    return df_ban_concatene

def load_dvf_gare(
    bucket_name: str,
    uri_dvf: str,
    uri_gare: str # URI unique pour les gares
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Charge les DataFrames : DVF (simple), BAN (concaténé), Gares (simple)
    depuis GCS.
    """
    print("--- Début du chargement des données dvf ---")
    bucket = storage_client.bucket(bucket_name)
    blob_dvf = bucket.blob(uri_dvf)


        # Ouvre le blob comme un fichier en mémoire et le passe à pandas
    with blob_dvf.open("rb") as f:
            df_dvf = pd.read_csv(f,sep=",")

    print(f"✅ Fichier chargé : {uri_dvf} ({len(df_dvf):,} lignes)")

    print("--- Début du chargement des données gare ---")
    bucket = storage_client.bucket(bucket_name)
    blob_gare = bucket.blob(uri_gare)


        # Ouvre le blob comme un fichier en mémoire et le passe à pandas
    with blob_gare.open("rb") as f:
            df_gare = pd.read_csv(f,sep=",")

    print(f"✅ Fichier chargé : {uri_dvf} ({len(df_gare):,} lignes)")







    return df_dvf, df_gare



def _load_single_csv_clean(bucket_name: str, uri_file: str) -> pd.DataFrame:
    """
    Fonction interne pour charger un fichier CSV unique depuis GCS.
    Elle gère la connexion, la lecture, et les erreurs.
    """
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(uri_file)


        # Ouvre le blob comme un fichier en mémoire et le passe à pandas
        with blob.open("rb") as f:
            df = pd.read_csv(f)

        print(f"✅ Fichier chargé : {uri_file} ({len(df):,} lignes)")
        return df

    except Exception as e:
        print(f"❌ ERREUR de chargement pour {uri_file}: {e}")
        # Retourne un DataFrame vide en cas d'échec pour ne pas bloquer le reste
        return pd.DataFrame()
