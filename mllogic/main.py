from load_data import load_dvf_gare, list_uris_in_bucket, load_ban,_load_single_csv
from clean_data import clean_data_transactions, clean_data_gares, clean_data_ban
from feature_engineering import run_feature_engineering
from train_test import train_test_split_strict_chrono
from assemble import assemble_dataframes
from encoder import preprocess_df
from model import model

# Import data
df_dvf, df_gare = load_dvf_gare(bucket_name='rail-estate-data', uri_dvf= 'dvf/dvf_idf_2014_2025.csv', uri_gare = 'gares/csv_gares.csv')
df_ban = load_ban(bucket_name='rail-estate-data', prefix_ban='ban/')
df_metro_clean = _load_single_csv(bucket_name='rail-estate-data',uri_file="gares/donnees_metro_clean2.csv")

# Clean data
df_dvf_clean = clean_data_transactions(df_dvf)
df_gare_clean = clean_data_gares(df_gare)
df_ban_clean = clean_data_ban(df_ban)
df_gare_merge_clean = assemble_dataframes(df_gare_clean,df_metro_clean)

# Feature engineering
df_merge_with_all_gare = run_feature_engineering(df_dvf_clean, df_gare_merge_clean, df_ban_clean)
df_merge_with_new_gare = run_feature_engineering(df_dvf_clean, df_gare_clean, df_ban_clean)


# Test train split Machine Learning
X_train, X_test, y_train, y_test = train_test_split_strict_chrono(df_merge_with_new_gare, date_col="annee", min_year=2014, max_year=2018, test_size = 0.2)

# Encoding & preprocessing
X_train_encoded = preprocess_df(X_train)
X_test_encoded = preprocess_df(X_test)

# Model 0
model, mae, rmse, r2, y_pred = model(X_train_encoded, X_test_encoded, y_train, y_test)
