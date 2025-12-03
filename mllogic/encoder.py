import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline

# Colonnes de ton Excel
cat_cols = ["Nature mutation","Type local"]
num_cols = [
    "Surface reelle bati",
    "prix_m2",
    "Nombre pieces principales",
    "lon",
    "lat",
    "distance_gare_km",
    "relative_year_signature",
    "relative_year_opening",
]

# ----- Définition du préprocesseur -----
preprocessor = ColumnTransformer(
    transformers=[
        # label encoding pour la colonne catégorielle
        (
            "cat",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",  # gère les nouvelles catégories
                unknown_value=-1
            ),
            cat_cols,
        ),
        # robust scaler pour les colonnes numériques
        ("num", RobustScaler(), num_cols),
        ("minmax", MinMaxScaler(), 'annee')
    ]
)

# Si tu veux l’intégrer dans un pipeline de modèle :
# from sklearn.linear_model import LinearRegression
# model = Pipeline(steps=[("preprocess", preprocessor),
#                        ("regressor", LinearRegression())])

# ----- Exemple d’utilisation seule du préprocesseur -----


def preprocess_df(df: pd.DataFrame):
    """Applique label encoding + robust scaler comme dans ton Excel."""
    X_trans = preprocessor.fit_transform(df)
    # On récupère un DataFrame propre avec les mêmes noms de colonnes
    cols_out = cat_cols + num_cols
    df_trans = pd.DataFrame(X_trans, columns=cols_out, index=df.index)
    return df_trans




# Exemple :
#df_prep = preprocess_df(df)
