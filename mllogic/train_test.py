from sklearn.model_selection import train_test_split


from sklearn.model_selection import train_test_split
import pandas as pd

def train_test_split_final(
    df: pd.DataFrame,
    min_year: int,
    max_year: int,
    test_size: float = 0.2,
    random_state: int = 42
):

    df= df[
        (df['annee'].dt.year >= min_year) &
        (df['annee'].dt.year <= max_year)
    ]
    y = df["prix_m2"] # Target (cible)
    X = df.drop(columns=["prix_m2"]) # Features (variables explicatives)

    # 3. Split Classique (Aléatoire)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    # 4. Retour des quatre jeux de données
    return X_train, X_test, y_train, y_test
