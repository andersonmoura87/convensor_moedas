"""Promove o modelo para stage Production no MLflow Registry."""

import mlflow

model_name = "exchange-rate-lstm"
client = mlflow.MlflowClient("http://localhost:5000")
versions = client.search_model_versions(f"name='{model_name}'")
if not versions:
    raise RuntimeError(f"No registered versions found for {model_name}.")

latest_version = max(versions, key=lambda version: int(version.version))
client.transition_model_version_stage(
    name=model_name,
    version=latest_version.version,
    stage="Production",
    archive_existing_versions=True,
)
print(f"Modelo '{model_name}' v{latest_version.version} promovido para Production")
