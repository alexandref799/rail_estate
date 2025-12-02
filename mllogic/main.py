from load_data import load_dvf_gare, list_uris_in_bucket, load_ban
from clean_data import clean_data_transactions, clean_data_gares, clean_data_ban
from feature_engineering import 
from encoder import preprocess_df
from model import model 

# Import data
df_dvf, df_gare = load_dvf_gare(bucket_name='rail-estate-data', uri_dvf= 'dvf/dvf_idf_2014_2025.csv', uri_gare = 'gares/csv_gares.csv')
df_ban = load_ban(bucket_name='rail-estate-data', prefix_ban='ban/')

# Clean data
df_dvf_clean = clean_data_transactions(df_dvf)
df_gare_clean = clean_data_gares(df_gare)
df_ban_clean = clean_data_ban(df_ban)

# Feature engineering
df_dvf_gps = feat(df_dvf_clean)
df_dvf_gps_gare = feat(df_dvf_gps)
df_dvf_gps_gare_relative_years = feat(df_dvf_gps_gare)
df_merge = merge(all datas)

# Encoding & preprocessing
df_encoded = preprocess_df(df_merge)

# Model 0
model, mae, rmse, r2, y_pred = model(df_encoded)
