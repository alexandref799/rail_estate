from fastapi import FastAPI
# from rail_estate.projet_rail_estate.ml_logic.model import load_model, predict  # à adapter

app = FastAPI()

@app.get("/")
def root():
    return {'greeting' : 'Hello'}

# # Charger le modèle une seule fois au démarrage
# app.state.model = load_model()

# @app.get("/")
# def root():
#     return {"greeting": "Hello Rail Estate"}

# @app.get("/predict")
# def predict_endpoint(
#     pickup_datetime: str,
#     pickup_longitude: float,
#     pickup_latitude: float,
#     dropoff_longitude: float,
#     dropoff_latitude: float,
#     passenger_count: int,
# ):
#     # adapter à tes features rail-estate
#     features = {
#         "pickup_datetime": pickup_datetime,
#         "pickup_longitude": pickup_longitude,
#         "pickup_latitude": pickup_latitude,
#         "dropoff_longitude": dropoff_longitude,
#         "dropoff_latitude": dropoff_latitude,
#         "passenger_count": passenger_count,
#     }
#     y_pred = predict(app.state.model, features)
#     return {"fare": float(y_pred)}
