import glob
import os
import time
import pickle
import keras
import locale
import joblib
from main import model

from params import *

# a ajouté cette fonction from chat gpt (

def _ensure_directories():

    for sub in ["params", "metrics", "models"]:
        os.makedirs(os.path.join(LOCAL_REGISTRY_PATH, sub), exist_ok=True)

# )

def save_results(params: dict, metrics: dict) -> None:

    _ensure_directories()

    # Save params locally
    timestamp = time.strftime("%Y%m%d")

    if params is not None:
        params_path = os.path.join(LOCAL_REGISTRY_PATH, "params", "_XGB_" +"R2" + " " + str(metrics["r2"]) + timestamp   + ".pickle")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

        # Save metrics locally

    if metrics is not None:
        metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", "_XGB_" "R2_" + " " + str(metrics["r2"]) + timestamp  + ".pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)


    print("✅ Results saved locally")


def save_info_model(model, model_name: str= "xgb_model") -> None:

    _ensure_directories()
    timestamp = time.strftime("%Y%m%d")

    # Save model locally
    model_path = os.path.join(
        LOCAL_REGISTRY_PATH,
        "models",
        "_XGB_" + f"{timestamp}.h5")

    model.save_model(model_path)

    print("✅ Model saved locally")




    return None
