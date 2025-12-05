import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

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

def train_test_ts(df:pd.DataFrame):
    def create_sequences(data, time_steps, target_index):
        X, y = [], []
        for i in range(len(data) - time_steps):
            X.append(data[i:(i + time_steps)])
            y.append(data[i + time_steps, target_index])
        return np.array(X), np.array(y)


# Initialisation des listes pour stocker toutes les séquences
    all_X, all_y = [], []

    # Traitement série par série (par gare)
    for gare in df[colonne_gare].unique():
        # Filtrer et supprimer la colonne texte de gare pour la conversion en array
        df_gare = df[df[colonne_gare] == gare].drop(columns=[colonne_gare])

        data_values = df_gare.values
        # print(df_gare.shape)
        # Trouver l'index de la cible dans l'array pour le séquençage
        target_index = df_gare.columns.get_loc(TARGET_COLUMN)

        # Créer les séquences
        X_data, y_data = create_sequences(data_values, TIME_STEPS, target_index)
        # print(len(X_data))
        # Ajouter à la liste globale
        all_X.append(X_data)
        all_y.append(y_data)

    # Concaténer toutes les séquences de toutes les gares

    X_data = np.concatenate(all_X)
    y_data = np.concatenate(all_y)



    # Division finale en ensembles d'entraînement et de test
    # Pour les séries temporelles, il est préférable d'utiliser une coupure temporelle ou,
    # comme ici, une coupe simple par proportion pour l'exemple.
    train_size = int(len(X_data) * 0.8)
    X_train, X_test = X_data[:train_size], X_data[train_size:]
    y_train, y_test = y_data[:train_size], y_data[train_size:]

    print(f"✅ Préparation des données terminée.")
    print(f"Forme de X_train (séquences d'entrée) : {X_train.shape}")
    print(f"Forme de y_train (cibles) : {y_train.shape}")

    return X_train,X_test,y_train,y_test
