from sklearn.ensemble import RandomForestClassifier
import mlflow
import utils
import os

mlflow.sklearn.autolog()
mlflow.set_experiment("random_forest")

X_train, y_train, X_test, y_test = utils.load_dataset()

with mlflow.start_run() as run:
    mlflow.set_experiment_tag("source", "github-action")
    
    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    utils.log_metrics(model, "testing", X_test, y_test, class_metric=True)
    
    mlflow.sklearn.save_model(model, "model")

    run_id = run.info.run_id
        
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            print(f"mlflow_run_id={run_id}", file=f)