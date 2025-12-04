import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV


def search_model_randomforestregressor(X_train, X_test, y_train, y_test, n_estimators= [500], max_depth= [6]):
    
    # Random forest model definition

    model = RandomForestRegressor(
        n_jobs=-1,  
    )
    
    #Paramgrid

    param_grid = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
    }
    
    # Implemeentation of the model & Param grid

    search_CV = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=50,
        cv=5,               # facultatif, par défaut = 5
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        random_state=42
)


    # 4. Train the model

    print("Training RandomForestRegressor...")
    
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
