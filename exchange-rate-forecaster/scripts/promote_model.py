"""Promove o modelo para stage Production no MLflow Registry."""
import mlflow

client = mlflow.MlflowClient("http://localhost:5000")
client.transition_model_version_stage("exchange-rate-lstm", 1, "Production")
print("Modelo 'exchange-rate-lstm' v1 promovido para Production")
