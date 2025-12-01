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
    
    #Fonction de normalization
     
    def normalize(s):
        if pd.isna(s):
            return ""
        s = str(s).upper().strip()
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")
        return s

    # Transform the address format
    
    mapping_type_voie = {
    'ALL': 'ALLEE', 'AV': 'AVENUE',
    'BD': 'BOULEVARD', 'BER': 'BERGE', 'BORD': 'BORD', 'CAR': 'CARREFOUR',
    'CC': 'CENTRE COMMERCIAL', 'CD': 'CHEMIN DEPARTEMENTAL', 'CHE': 'CHEMIN',
    'CHEM': 'CHEMIN', 'CHS': 'CHAUSSEE', 'CHT': 'CHALET', 'CHV': 'CHEMIN VICINAL',
    'CITE': 'CITE', 'CLOS': 'CLOS', 'COTE': 'COTE', 'COUR': 'COUR', 'CR': 'CHEMIN RURAL',
    'CRS': 'COURS', 'CRX': 'CROIX', 'CTR': 'CENTRE', 'CTRE': 'CENTRE', 'D': 'DOMAINE',
    'DOM': 'DOMAINE', 'ESC': 'ESCALIER', 'ESP': 'ESPLANADE',
    'FG': 'FAUBOURG', 'FRM': 'FERME', 'GAL': 'GALERIE', 'GPL': 'GROUPE D IMMEUBLES',
    'GR': 'GRANDE RUE', 'HAM': 'HAMEAU', 'HLM': 'HABITATION A LOYER MODERE',
    'IMP': 'IMPASSE', 'JARD': 'JARDIN', 'LOT': 'LOTISSEMENT', 'MAIL': 'MAIL',
    'MTE': 'MONTEE',
    'PARC': 'PARC', 'PAS': 'PASSAGE', 'PASS': 'PASSAGE', 'PCH': 'PETITE CHAUSEE',
    'PKG': 'PARKING', 'PL': 'PLACE', 'PLA': 'PLATEAU', 'PLE': 'PETITE LEVEE',
    'PONT': 'PONT', 'PORT': 'PORT', 'PROM': 'PROMENADE', 'PRV': 'PARVIS',
    'PTE': 'PORTE', 'PTR': 'PETITE ROUTE', 'PTTE': 'PETITE', 'QUA': 'QUARTIER',
    'QUAI': 'QUAI', 'RES': 'RESIDENCE', 'RLE': 'RUELLE', 'ROC': 'ROCADE',
    'RPE': 'RAMPE', 'RPT': 'ROND-POINT', 'RTD': 'ROUTE DEPARTEMENTALE',
    'RTE': 'ROUTE', 'RUE': 'RUE', 'SEN': 'SENTE', 'SQ': 'SQUARE',
    'TRA': 'TRAVERSE', 'TSSE': 'TERRASSE', 'VAL': 'VAL', 'VALL': 'VALLEE',
    'VC': 'VOIE COMMUNALE', 'VCHE': 'VIEILLE CHEMIN', 'VEN': 'VENELLE',
    'VGE': 'VILLAGE', 'VIL': 'VILLE', 'VLA': 'VILLA', 'VOIE': 'VOIE',
    'VOIR': 'VOIRIE', 'VTE': 'VENTE', 'ZA': 'ZA','ZAC': 'ZAC'}
    
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
    "Code commune",
    "Prefixe de section",
    "Section",
    "No plan"]

    df_addr = df[cols_to_keep].copy()

    # type voie complet
    df_addr["type_voie_expanded"] = df_addr["Type de voie"].map(mapping_type_voie).fillna(df_addr["Type de voie"])

    # numero DVF
    df_addr["numero"] = (
        df_addr["No voie"]
        .fillna("")
        .astype(str)
        .str.replace(".0", "", regex=False)
    )

    # voie DVF normalisée
    df_addr["voie_clean"] = (df_addr["type_voie_expanded"].fillna("") 
                             + " " 
                             + df_addr["Voie"].fillna("")).apply(normalize)

    # code postal DVF
    df_addr["code_postal"] = (
        df_addr["Code postal"]
        .astype(str)
        .str.replace(".0", "", regex=False)
        .str[:5]
    )
    
    # Chargement fichier BAN (départements IDF)
    
    ban_files = glob.glob("/Users/alexandreferreira/Desktop/projet_dvf/data/ban/*.csv.gz")  # Comment bien gérer ça ?

    ban_list = []
    for file in ban_files:
        print("Loading:", file)
        df_ban = pd.read_csv(
            file,
            compression="gzip",
            sep=";",
            low_memory=False,
            dtype={"code_postal": str, "numero": str}
        )
        ban_list.append(df_ban)

    ban = pd.concat(ban_list, ignore_index=True)

    # normalisation BAN
    ban["numero"] = ban["numero"].fillna("").astype(str).str.replace(".0", "", regex=False)
    ban["voie_clean"] = ban["nom_voie"].apply(normalize)
    ban["code_postal"] = ban["code_postal"].astype(str).str[:5]

    # rename lon/lat
    ban = ban.rename(columns={"lon": "longitude", "lat": "latitude"})

    # sélectionner colonnes utiles
    ban = ban[["numero", "voie_clean", "code_postal", "longitude", "latitude"]]
    
    # Merge DVP et BAN (GPS)
    df = df_addr.merge(
    ban,
    how="left",
    on=["numero", "voie_clean", "code_postal"]
    )

    print("Shape df_geo:", df.shape)
    print("Missing GPS ratio:", df[["longitude","latitude"]].isna().mean())

    
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
    
    return df
