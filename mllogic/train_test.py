#Je veux train test split en gardant un aspect chronologique
from sklearn.model_selection import train_test_split

def train_test_split_chrono(df, date_col, test_size=0.2):
    # Trier le DataFrame par la colonne de date
    df_sorted = df.sort_values(by=date_col)

    # Calculer l'index de séparation
    split_index = int(len(df_sorted) * (1 - test_size))

    # Diviser le DataFrame en ensembles d'entraînement et de test
    train_df = df_sorted.iloc[:split_index]
    test_df = df_sorted.iloc[split_index:]

    return train_df, test_df
# Exemple d'utilisation :
# train_df, test_df = train_test_split_chrono(df, date_col='date_column


