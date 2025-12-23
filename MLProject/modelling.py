from sklearn.ensemble import RandomForestClassifier
import mlflow
import utils

mlflow.sklearn.autolog()
mlflow.set_experiment("random_forest")

X_train, y_train, X_test, y_test = utils.load_dataset()

with mlflow.start_run():
    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)

    utils.log_metrics(model, "testing", X_test, y_test, class_metric=True)