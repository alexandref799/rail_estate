from load_data import load_dvf_gare, list_uris_in_bucket, load_ban,_load_single_csv
from clean_data import clean_data_transactions, clean_data_gares, clean_data_ban
from feature_engineering import run_feature_engineering
from train_test import train_test_split_strict_chrono
from assemble import assemble_dataframes
from encoder import preprocess_df
from model_xgbregressor import search_xgbregressor
from model_randomforest import search_model_randomforestregressor
from model_gradientboosting import search_gbregressor

'''
model_name = [xgb_]'''
def train(model_name, all_gares):

    # Import data
    df_dvf, df_gare = load_dvf_gare(bucket_name='rail-estate-data', uri_dvf= 'dvf/dvf_idf_2014_2025.csv', uri_gare = 'gares/csv_gares.csv')
    df_ban = load_ban(bucket_name='rail-estate-data', prefix_ban='ban/')
    df_taux = _load_single_csv(bucket_name='rail-estate-data', uri_file='taux/data_taux.csv') #déjà clean
    df_insee = _load_single_csv(bucket_name="rail-estate-data", uri_file="insee/data_insee.csv") #déjà clean

    # Clean data
    df_dvf_clean = clean_data_transactions(df_dvf)
    df_ban_clean = clean_data_ban(df_ban)
    df_gare_clean = clean_data_gares(df_gare)

    if all_gares == 0:
        #Seulement les nouvelles gares
        df_merge_with_new_gare = run_feature_engineering(df_dvf_clean, df_gare_clean, df_ban_clean, df_taux, df_insee)
        df_merge = df_merge_with_new_gare.copy()

    elif all_gares == 1:
        # Import des anciens metros
        df_metro_clean = _load_single_csv(bucket_name='rail-estate-data',uri_file="gares/donnees_metro_clean2.csv")

        #Toutes les gares sont présentes
        df_gare_merge_clean = assemble_dataframes(df_gare_clean,df_metro_clean)
        df_merge_with_all_gare = run_feature_engineering(df_dvf_clean, df_gare_merge_clean, df_ban_clean, df_taux)
        df_merge = df_merge_with_all_gare.copy()

    # Test train split chronologique
    X_train, X_test, y_train, y_test = train_test_split_strict_chrono(df_merge, date_col="annee", min_year=2014, max_year=2020, test_size = 0.2)

    # Encoding & preprocessing
    X_train_encoded = preprocess_df(X_train)
    X_test_encoded = preprocess_df(X_test)

    # Model
    if model_name == 'ml-xgb':
        best_model, results_model, best_params, y_pred = search_xgbregressor(X_train_encoded, X_test_encoded, y_train, y_test,
                                                                                           n_estimators= [500], learning_rate=[0.05], max_depth= [8], subsample = [0.7], colsample= [0.8])
    elif model_name == 'ml_rfr':
        best_model, results_model, best_params, y_pred = search_xgbregressor(X_train_encoded, X_test_encoded, y_train, y_test,
                                                                                           n_estimators= [500], learning_rate=[0.05], max_depth= [8], subsample = [0.7], colsample= [0.8])
    elif model_name == 'ml_gbr':
        best_model, results_model, best_params, y_pred = search_gbregressor(X_train_encoded, X_test_encoded, y_train, y_test, n_estimators=[200], learning_rate=[0.1], max_depth=[5], subsample=[0.7], colsample=[0.8], random_state=42, n_iter=50, cv=5)

    return best_model, results_model, best_params, y_pred
