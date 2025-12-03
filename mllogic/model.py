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



def search(X_train, X_test, y_train, y_test, n_estimators= [500], learning_rate=[0.05], max_depth= [8], subsample = [0.7], colsample= [0.8] ):

    # 3. XGBoost model definition

    model = XGBRegressor(
        objective="reg:squarederror",
        n_jobs=-1,
        tree_method="hist",   # fast on CPU
    )
    #Paramgrid

    param_grid = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample,
}
    #implemeentation of the model & Param grid

    search_CV = RandomizedSearchCV(
        model,
        param_grid
        )

    # ---------------------------------------------------
    # 4. Train the model

    print("Training XGBoost...")
    Fited_search = search_CV.fit(X_train, y_train)


    best_model = Fited_search.best_estimator_


    # 5. Evaluation

    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results_model = {"mae": mae, "rmse": rmse, "r2": r2}

    print("\n=== Evaluation Metrics ===")
    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²  : {r2:.3f}")


    return best_model, results_model, Fited_search.best_params_, y_pred
