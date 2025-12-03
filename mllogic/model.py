import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
from sklearn.model_selection import RandomizedSearchCV

# 1. Load your engineered dataframe
# ---------------------------------------------------

def model(X_train, X_test, y_train, y_test):

    from mllogic.encoder import preprocess_df
    import pandas as pd



    # ---------------------------------------------------
    # 2. Train/test split
    # ---------------------------------------------------

#X_train, X_test, y_train, y_test = train_test_split(
#X, y,
#test_size=0.2,
#random_state=42,)

def model(X_train, X_test, y_train, y_test):

    # 3. XGBoost model definition

    model = XGBRegressor(
        objective="reg:squarederror",
        n_jobs=-1,
        tree_method="hist",   # fast on CPU
    )


    param_grid = {
        "n_estimators": [200, 500, 800],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [4, 6, 8],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
}

    search = RandomizedSearchCV(
        model,
        param_grid
        )

    # ---------------------------------------------------
    # 4. Train the model

    print("Training XGBoost...")
    search.fit(X_train, y_train)


    # 5. Evaluation

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n=== Evaluation Metrics ===")
    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²  : {r2:.3f}")


    return model, mae, rmse, r2, y_pred
