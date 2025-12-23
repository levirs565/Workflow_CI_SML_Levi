import mlflow
import pandas as pd
import os
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay, roc_auc_score, log_loss, precision_score, recall_score, f1_score, accuracy_score
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_REPO = os.getenv("DAGSHUB_REPO")
if DAGSHUB_USERNAME is not None and DAGSHUB_REPO is not None:
    import dagshub
    dagshub.init(repo_owner=DAGSHUB_USERNAME,
                 repo_name=DAGSHUB_REPO, mlflow=True)
elif os.getenv("MLFLOW_TRACKING_URI") is None:
    print("Please set either DAGSHUB_USERNAME and DAGSHUB_REPO or MLFLOW_TRACKING_URI")
    exit(1)

CURRENT_DIR = Path(__file__).resolve().parent
PREPROCESSED_DIR = CURRENT_DIR / "weather_preprocessing"


def split_dataset(df):
    return df.drop(columns=["RainTomorrow"]), df["RainTomorrow"]


def load_dataset():
    X_train, y_train = split_dataset(
        pd.read_csv(PREPROCESSED_DIR / "train.csv"))
    X_test, y_test = split_dataset(pd.read_csv(PREPROCESSED_DIR / "test.csv"))
    return X_train, y_train, X_test, y_test


target_names = ["not_rain", "rain"]


def log_metrics(model, prefix, X, y, class_metric=False):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    mlflow.log_metric(f"{prefix}_precision_score",
                      precision_score(y, y_pred, average="weighted"))
    mlflow.log_metric(f"{prefix}_recall_score",
                      recall_score(y, y_pred, average="weighted"))
    mlflow.log_metric(f"{prefix}_f1_score", f1_score(
        y, y_pred, average="weighted"))
    mlflow.log_metric(f"{prefix}_accuracy_score", accuracy_score(y, y_pred))
    mlflow.log_metric(f"{prefix}_log_loss", log_loss(y, y_prob))
    mlflow.log_metric(f"{prefix}_roc_auc", roc_auc_score(
        y, y_prob, average="weighted"))
    mlflow.log_metric(f"{prefix}_score", model.score(X, y))

    confusion_display = ConfusionMatrixDisplay.from_predictions(
        y, y_pred, display_labels=target_names, normalize="true")
    mlflow.log_figure(confusion_display.figure_,
                      f"{prefix}_confusion_matrix.png")

    roc_display = RocCurveDisplay.from_predictions(y, y_prob)
    mlflow.log_figure(roc_display.figure_, f"{prefix}_roc_curve.png")

    precision_recall_display = PrecisionRecallDisplay.from_predictions(
        y, y_prob)
    mlflow.log_figure(precision_recall_display.figure_,
                      f"{prefix}_precision_recall_curve.png")

    report_text = classification_report(y, y_pred, target_names=target_names)
    mlflow.log_text(report_text, f"{prefix}_classification_report.txt")

    if class_metric:
        report = classification_report(
            y, y_pred, output_dict=True, target_names=target_names)

        for key in target_names:
            data = report[key]
            for metric_name, value in data.items():
                mlflow.log_metric(f"test_{metric_name}_{key}", value)
