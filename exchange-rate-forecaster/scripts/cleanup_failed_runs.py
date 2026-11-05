"""Remove runs com status FAILED do experimento no MLflow."""
import mlflow

client = mlflow.MlflowClient("http://localhost:5000")
experiment = client.get_experiment_by_name("exchange-rate-lstm")

if not experiment:
    print("Experimento não encontrado.")
    exit()

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="status = 'FAILED'",
)

if not runs:
    print("Nenhum run falho encontrado.")
    exit()

for run in runs:
    client.delete_run(run.info.run_id)
    print(f"Run removido: {run.info.run_id}")

print(f"Total removido: {len(runs)} run(s)")
