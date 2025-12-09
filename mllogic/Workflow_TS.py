import pandas as pd
from mllogic.Utils_TS import group_by,split_train_test,create_sequences,create_sequences_multi_horizon,Config
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pandas as pd, numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler
# --- 0. DÉFINITION DES VARIABLES (À VÉRIFIER) ---
# Assurez-vous que ces noms de colonnes correspondent EXACTEMENT à ceux de votre DataFrame
colonne_date = 'date'      # Le nom de votre colonne de date/heure
colonne_gare = 'nom_gare'  # Le nom de votre colonne d'identifiant de gare

colonnes_numeriques = ['Surface reelle bati', 'prix_m2', 'Nombre pieces principales',
                       'distance_gare_km', 'relative_signature', 'relative_opening', 'taux','lat_gare',
            'lon_gare',
            'variation_mom',
            'variation_yoy',
            'Revenu médian 2021',
            'Évol. annuelle moy. de la population 2016-2022',
            'Part proprio en rés. principales 2022',
            'Part locataires HLM dans les rés. principales 2022',
            'Part cadres sup. 2022', 'Part logements vacants 2022',
            'Taux de chômage annuel moyen 2024',
            'Part des élèves du privé parmi les élèves du second degré 2024',
            "Nombre d'établissements 2023"]
colonnes_categorielles = ['Nature mutation', 'Type local']

TARGET_COLUMN = 'prix_m2' # La colonne que le LSTM va prédire
TIME_STEPS = 6


# Load merged dataframe
csv_path = "df_merge_with_all_gare.csv"  # adapt if needed

df_merge_with_all_gare = pd.read_csv(csv_path)
print("Loaded df_merge_with_all_gare:", df_merge_with_all_gare.shape)
#A modifier avec la fonction load_single CSV

df_group = group_by(df_merge_with_all_gare,
                    colonne_date=colonne_date,
                    colonne_gare=colonne_gare,
                    colonnes_numeriques=colonnes_numeriques,
                    colonnes_categorielles=colonnes_categorielles)


# Add a stable Gare_ID mapping
# factorize returns (codes, uniques); we keep both for later recovery
codes, uniques = pd.factorize(df_group[colonne_gare])
df_group = df_group.copy()
df_group["Gare_ID"] = codes

# Mapping helpers
gare_id_to_name = dict(enumerate(uniques))
gare_name_to_id = {name: idx for idx, name in gare_id_to_name.items()}

#df_issy = df_group[df_group['nom_gare']=='Issy']
#Possible sélection de gare

df_train, df_test = split_train_test(df_group,cfg=Config,test_start='2024-04-01',test_end='2025-06-01')


df_train = df_train.drop(columns='nom_gare')
df_test = df_test.drop(columns='nom_gare')
# a mettre en  fonction

#Scale Data

X_scaler= RobustScaler()

X_train = X_scaler.fit_transform(df_train)
X_test  = X_scaler.transform(df_test)

y_scaler = MinMaxScaler()

y_train = y_scaler.fit_transform(df_train[[TARGET_COLUMN]])
y_test  = y_scaler.transform(df_test[[TARGET_COLUMN]])

n_steps =12

X_train_seq, y_train_seq = create_sequences(
    X_train, y_train, n_steps
)
X_test_seq, y_test_seq = create_sequences(
    X_test, y_test, n_steps
)



model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(n_steps, X_train_seq.shape[2])),
    LSTM(32),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)


early_stop = EarlyStopping(
    monitor='val_loss',        # métrique surveillée
    patience=10,               # nb d'epochs sans amélioration avant d'arrêter
    mode='min',                # on cherche à minimiser la loss
    restore_best_weights=True  # recharge les poids du meilleur epoch
)

history = model.fit(
    X_train_seq,
    y_train_seq,
    validation_split=0.2,   # prend les 20% de fin comme validation
    epochs=200,
    batch_size=32,
    shuffle=False,
    callbacks=[early_stop],
    verbose=2
)



pd.DataFrame(history.history)[["loss","val_loss"]].plot()

test_loss = model.evaluate(X_test_seq, y_test_seq, verbose=0)

y_pred_scaled = model.predict(X_test_seq)
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_true = y_scaler.inverse_transform(y_test_seq.reshape(-1, 1))

mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(f"Test loss (MSE scaled): {test_loss:.4f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
