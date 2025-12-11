# backend/app/services/ml/api.py
from backend.app.services.ml.train_demand_model import train_all
from backend.app.services.predict_next_day import predict_next_day

def retrain_and_forecast():
    """
    Trainiert alle Modelle mit den aktuellsten Daten
    und gibt direkt die neue Vorhersage als DataFrame zurück.
    """
    train_all()
    forecast = predict_next_day()
    return forecast
