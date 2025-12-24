from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score
import mlflow
import utils
import os

mlflow.sklearn.autolog()
mlflow.set_experiment("random_forest")

X_train, y_train, X_test, y_test = utils.load_dataset()

with mlflow.start_run() as run:
    mlflow.set_experiment_tag("source", "github-action")
    
    params = {
            'n_estimators':300,
            'max_depth': 10,
            'min_samples_split': 7,
            'max_features':  0.2700686061348578,
            'min_samples_leaf': 4,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
    }
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    utils.log_metrics(model, "testing", X_test, y_test, class_metric=True)
    
    y_pred = model.predict(X_test)
    score = fbeta_score(y_test, y_pred, beta=2)
    mlflow.log_metric("test_f2_score_rain", score)

    mlflow.sklearn.save_model(model, "model")

    run_id = run.info.run_id
        
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            print(f"mlflow_run_id={run_id}", file=f)