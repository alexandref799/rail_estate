from datetime import datetime
import pandas as pd
from load_data import _load_single_csv_clean
from Utils_TS import group_by2,split_train_test,make_sequences_nico
from config import Config
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler
from load_model import load_model_from_gcp
from plot_forecast import plot_predictions
import joblib
from pathlib import Path



def pred_selector(
    bucket_name: str = "rail-estate-data",
    uri_csv: str = "clean_data/df_merge_with_all_gare.csv",
    is_new_gare: str | None = None,
    max_distance_km: float | None = None,
    gare_name: str | None = None,
    min_transac: float | None = None,
    blob_name = 'models/model_long',
    end_date : str = '2024-12-01',
) -> pd.DataFrame:
    """
    Charge le dataframe et applique des filtres simples.

    Args:
        bucket_name: Nom du bucket GCS.
        uri_file: Chemin du fichier CSV dans le bucket.
        is_new_gare: "new" pour is_new_gare=1, "old" pour is_new_gare=0, None pour ne pas filtrer.
        max_distance_km: Si renseigné, garde les lignes avec distance_gare_km <= cette valeur.

    Returns:
        DataFrame filtré.
    """

    # Paths and basic configs
    csv_path = "df_merge_with_all_gare.csv"  # adapt if needed
    col_date = "date"
    col_gare = "nom_gare"
    target_col = "prix_m2"

    num_cols = [
        'Surface reelle bati', 'Nombre pieces principales',
        'distance_gare_km', 'relative_signature', 'relative_opening', 'taux',
        'lat_gare', 'lon_gare', 'variation_mom', 'variation_yoy',
        'Revenu médian 2021', 'Évol. annuelle moy. de la population 2016-2022',
        'Part proprio en rés. principales 2022',
        'Part locataires HLM dans les rés. principales 2022',
        'Part cadres sup. 2022', 'Part logements vacants 2022',
        'Taux de chômage annuel moyen 2024',
        'Part des élèves du privé parmi les élèves du second degré 2024',
        "Nombre d'établissements 2023"
    ]

    cat_cols = ['Nature mutation', 'Type local']

    n_steps = 18
    horizon = 6
    cache_path = Path('df_merge_with_all_gare.csv')

    if cache_path.is_file():
        df = pd.read_csv(cache_path, sep=',')
    else:
        df = _load_single_csv_clean(bucket_name=bucket_name, uri_file=uri_csv)

    if is_new_gare == "new":
        df = df[df["is_new_gare"] == 1]
    elif is_new_gare == "old":
        df = df[df["is_new_gare"] == 0]

    if max_distance_km is not None and "distance_gare_km" in df.columns:
        df = df[df["distance_gare_km"] <= float(max_distance_km)]

    if gare_name is not None and "nom_gare" in df.columns:
        df = df[df['nom_gare'] == str(gare_name)]

    df_group = group_by2(df)



    num_cols.append("share_appartement")
    num_cols.append("share_vente")


    if min_transac is not None and "n_transactions" in df_group.columns:
        df_filtered = df_group[df_group["n_transactions"] >= float(min_transac)]


    end_date = pd.to_datetime(end_date)
    start_date = end_date -pd.DateOffset(months=n_steps+horizon)

    # Sélection stricte entre les deux dates (inclusives) sur l'index date
    if not isinstance(df_filtered.index, pd.DatetimeIndex):
        if "date" in df_filtered.columns:
            df_filtered = df_filtered.set_index("date")
        else:
            raise ValueError("No datetime index or 'date' column available for date filtering.")


    df_filtered = df_filtered.sort_index()
    df_pred = df_filtered.loc[(df_filtered.index >= start_date) & (df_filtered.index <= end_date)]

    print(df_pred.columns)

    scaler = joblib.load('scalers/scaler1.joblib')

    df_pred[num_cols] = scaler.transform(df_pred[num_cols])
    print(df_pred.shape)
    print(len(df_pred.columns))

    ID_COL = 'nom_gare'
    DATE_COL = "date"
    TARGET_COL ='prix_m2'

    y_true = df_filtered['prix_m2'].loc[end_date -pd.DateOffset(months=horizon):end_date]


    min_rows = n_steps + horizon
    if len(df_pred) < min_rows:
        raise ValueError(
            f"Not enough rows in df_pred ({len(df_pred)}) for n_steps={n_steps} + horizon={horizon} (need >= {min_rows})."
        )
    X_seq, y_seq = make_sequences_nico(df_pred,seq_len=n_steps,horizon=horizon)

    print(X_seq.shape)

    if len(X_seq) == 0:
        raise ValueError("No sequences built (likely not enough rows for the chosen n_steps/horizon in this date window).")
    print(X_seq.shape, y_seq.shape)

    model = load_model_from_gcp(bucket_name=bucket_name,blob_name=blob_name,compile=False)
    print(model.input_shape)
    y_pred = model.predict(X_seq)

    # df_forecast = plot_predictions()

    return y_pred, y_true,
