import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

# ---------------------------------------------------
# 1. Load your engineered dataframe
# ---------------------------------------------------

def model(encoded_data):

    from mllogic.encoder import preprocess_df
    import pandas as pd

    # url = "mllogic/Output feature engineering - Copie de feat_eng.csv"

    # data_engineered = pd.read_csv(url)

    # encoded_data = preprocess_df(data_engineered)

    # encoded_data


    # Target
    y = encoded_data["prix_m2"]

    # Features = all columns except target
    X = encoded_data.drop(columns=["prix_m2"])


    # ---------------------------------------------------
    # 2. Train/test split
    # ---------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
    )


    # ---------------------------------------------------
    # 3. XGBoost model definition
    # ---------------------------------------------------

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        n_jobs=-1,
        tree_method="hist",   # fast on CPU
    )


    # ---------------------------------------------------
    # 4. Train the model
    # ---------------------------------------------------

    print("Training XGBoost...")
    model.fit(X_train, y_train)


    # ---------------------------------------------------
    # 5. Evaluation
    # ---------------------------------------------------

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n=== Evaluation Metrics ===")
    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²  : {r2:.3f}")


    return model, mae, rmse, r2, y_pred
