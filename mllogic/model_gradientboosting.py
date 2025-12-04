import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def search_gbregressor(
    X_train, X_test, y_train, y_test,
    n_estimators=[500],
    learning_rate=[0.05],
    max_depth=[8],
    subsample=[0.7],
    colsample=[0.8],           # équivalent de colsample_bytree -> map vers max_features
    random_state=42,
    n_iter=50,
    cv=5,
):
    """
    Randomized search pour GradientBoostingRegressor.
    Retourne (best_model, results_dict, best_params, y_pred)
    """

    # 1) Définition du modèle de base
    model = GradientBoostingRegressor(
        loss="squared_error",
        random_state=random_state
    )

    # 2) Param grid (RandomizedSearchCV accepte des listes comme distributions)
    #    mappe `colsample` en `max_features` (valeurs entre 0 et 1 acceptées)
    param_grid = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "subsample": subsample,
        "max_features": colsample,
    }

    # 3) RandomizedSearchCV
    search_cv = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        random_state=random_state,
        verbose=1
    )

    # 4) Entraînement
    print("Training GradientBoostingRegressor (RandomizedSearchCV)...")
    fitted_search = search_cv.fit(X_train, y_train)
    best_model = fitted_search.best_estimator_
    best_params = fitted_search.best_params_

    print("\nBest params found:")
    print(best_params)

    # 5) Évaluation
    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)  # sqrt(MSE)
    r2 = r2_score(y_test, y_pred)

    results_model = {"mae": mae, "rmse": rmse, "r2": r2}

    print("\n=== Evaluation Metrics ===")
    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²  : {r2:.3f}")

    return best_model, results_model, best_params, y_pred
