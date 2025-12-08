import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

# --- 0. DÉFINITION DES VARIABLES (À VÉRIFIER) ---
# Assurez-vous que ces noms de colonnes correspondent EXACTEMENT à ceux de votre DataFrame
colonne_date = 'date'      # Le nom de votre colonne de date/heure
colonne_gare = 'nom_gare'  # Le nom de votre colonne d'identifiant de gare

colonnes_numeriques = ['Surface reelle bati', 'prix_m2', 'Nombre pieces principales',
                       'distance_gare_km', 'relative_signature', 'relative_opening', 'taux']
colonnes_categorielles = ['Nature mutation', 'Type local']

colonne_date = 'date'      # Le nom de votre colonne de date/heure
colonne_gare = 'nom_gare'  # Le nom de votre colonne d'identifiant de gare

colonnes_numeriques = ['Surface reelle bati', 'prix_m2', 'Nombre pieces principales',
                       'distance_gare_km', 'relative_signature', 'relative_opening', 'taux']
colonnes_categorielles = ['Nature mutation', 'Type local']

TARGET_COLUMN = 'prix_m2' # La colonne que le LSTM va prédire
TIME_STEPS = 6            # Nombre de mois passés à utiliser pour la prédiction

def group_by(df:pd.DataFrame):

    df['date'] = pd.to_datetime(df['date'])
    df['Annee_Mois'] = df[colonne_date].dt.to_period('M')
    aggregation_functions = {}
    for col in colonnes_numeriques:
        aggregation_functions[col] = 'mean'
    for col in colonnes_categorielles:
        # Utilisation de np.nan pour les cas où le mode est vide
        aggregation_functions[col] = lambda x: x.mode()[0] if not x.mode().empty else np.nan

    # Application du GroupBy
    df_agg = df.groupby(['Annee_Mois', colonne_gare]).agg(aggregation_functions).reset_index()

    # Préparation de l'index temporel
    df_agg['date'] = df_agg['Annee_Mois'].dt.to_timestamp()
    df_agg = df_agg.drop(columns=['Annee_Mois'])
    df_agg = df_agg.set_index('date')


    # --- 2. SÉLECTION ET NETTOYAGE ---

    # Sélection des colonnes pertinentes
    colonnes_a_garder = [colonne_gare] + colonnes_numeriques + colonnes_categorielles
    df_final = df_agg[colonnes_a_garder].copy()

    # Suppression des lignes avec des valeurs manquantes (NaN)
    df_final = df_final.dropna()
    return df_final


def encoding(df:pd.DataFrame):
    # a) Label Encoding pour la colonne de Gare (pour le filtrage)
    le = LabelEncoder()
    df_encoded = df.copy()

    df_encoded['Nature mutation'] = le.fit_transform(df_encoded['Nature mutation'])
    df_encoded['Type local'] = le.fit_transform(df_encoded['Type local'])

    # b) One-Hot Encoding pour les autres colonnes catégorielle
    ohe = OneHotEncoder(drop='if_binary',sparse_output=False)
    df_encoded[ohe.get_feature_names_out()] =  ohe.fit_transform(df_encoded[[colonne_gare]])

    # df_encoded = pd.get_dummies(df_encoded, columns=colonnes_categorielles, drop_first=False)

    # c) Normalisation des caractéristiques (scaler tout ce qui est numérique)
    colonnes_a_normaliser = df_encoded.select_dtypes(include=['number']).columns.tolist()

    scaler = MinMaxScaler(feature_range=(0, 1))
    df_encoded[colonnes_a_normaliser] = scaler.fit_transform(df_encoded[colonnes_a_normaliser])

    return df_encoded
