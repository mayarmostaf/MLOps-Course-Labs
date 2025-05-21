import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "top_churn_models"
TRACKING_URI = "http://localhost:5000"

mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

# Get all experiment IDs
experiments = mlflow.search_experiments()
experiment_ids = [exp.experiment_id for exp in experiments]

# Search all runs by accuracy, no filter for NULL values
all_runs = mlflow.search_runs(
    experiment_ids=experiment_ids,
    filter_string="attributes.status = 'FINISHED'",
    order_by=["metrics.accuracy DESC"],
    max_results=2,
)

# Register top 2 models
for idx, (_, run) in enumerate(all_runs.iterrows()):
    run_id = run["run_id"]
    model_uri = f"runs:/{run_id}/model"

    # Register model
    model_details = mlflow.register_model(model_uri, MODEL_NAME)
    version = model_details.version
    print(f"✅ Registered: Version {version} from Run ID {run_id}")

    # Transition stage (Production for the top model, Staging for the second-best)
    stage = "Production" if idx == 0 else "Staging"
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=version,
        stage=stage,
        archive_existing_versions=True
    )
    print(f"🚀 Model version {version} transitioned to stage: {stage}")
