import os

import os
import json
import datetime
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


RESULTS_DIR = "../results/"
MODEL_DIR = "../models/saved_models/"
CM_DIR = os.path.join(RESULTS_DIR, "confusion_matrixes")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
CM_JSON_DIR = os.path.join(REPORTS_DIR, "confusion_matrix_jsons")
CR_JSON_DIR = os.path.join(REPORTS_DIR, "classification_reports_jsons")
os.makedirs(CM_JSON_DIR, exist_ok=True)
os.makedirs(CR_JSON_DIR, exist_ok=True)

os.makedirs(CM_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def log_basic_evaluation_results(test_acc, test_loss, model_name):
    """
    Logs and saves basic evaluation metrics (accuracy and loss).
    """
    print(f"📊 Evaluating model '{model_name}' on test set...")
    print(f"✅ Test Accuracy: {test_acc * 100:.2f}%")
    print(f"✅ Test Loss: {test_loss:.4f}")

    results = {"Test Accuracy": float(test_acc), "Test Loss": float(test_loss)}
    result_path = os.path.join(REPORTS_DIR, f"{model_name}_evaluation.json")
    # with open(result_path, 'w') as f:
    #     json.dump(results, f, indent=4)

    #print(f"📁 Evaluation results saved to: {result_path}")


def plot_classification_report(report_str : str, model_name: str):
    """
    Plots the classification report string as a matplotlib figure.
    """
    plt.figure(figsize=(10, 6))
    plt.axis("off")
    plt.title(f"Classification Report - {model_name}", fontsize=14, weight='bold', pad=20)
    plt.text(0.01, 0.95, report_str, family='monospace', fontsize=10)
    plt.tight_layout()

    fig_path = os.path.join(REPORTS_DIR, f"{model_name}_classification_report.png")
    plt.savefig(fig_path)
    plt.show()
    print(f"🖼️ Classification report figure saved to {fig_path}")
    report_txt_path = os.path.join(REPORTS_DIR, f"{model_name}_classification_report.txt")
    with open(report_txt_path, 'w') as f:
        f.write(report_str)
    print(f"📁 Classification report saved to: {report_txt_path}")


def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Plots and saves the confusion matrix for the model.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", xticks_rotation="vertical")
    plt.title(f"Confusion Matrix - {model_name}")

    cm_path = os.path.join(CM_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(cm_path)
    plt.show()

    print(f"🖼️ Confusion matrix saved to {cm_path}")

def _get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def save_classification_report_json(report_dict, model_name):
    timestamp = _get_timestamp()
    path = os.path.join(CR_JSON_DIR, f"{model_name}_classification_report.json")

    if os.path.exists(path):
        with open(path, 'r') as f:
            existing = json.load(f)
    else:
        existing = {}

    existing[timestamp] = report_dict

    with open(path, 'w') as f:
        json.dump(existing, f, indent=4)

    print(f"📝 Classification report saved to {path}")


def save_confusion_matrix_json(cm, model_name):
    timestamp = _get_timestamp()
    path = os.path.join(CM_JSON_DIR, f"{model_name}_confusion_matrix.json")

    if os.path.exists(path):
        with open(path, 'r') as f:
            existing = json.load(f)
    else:
        existing = {}

    existing[timestamp] = cm.tolist()

    with open(path, 'w') as f:
        json.dump(existing, f, indent=4)

    print(f"📝 Confusion matrix saved to {path}")

def save_overall_metrics(test_acc, test_loss, model_name, dataset_name="default"):
    """
    Saves test accuracy and loss with timestamp, tagged by dataset name.
    """
    RESULTS_DIR = "../results/reports/"
    METRICS_PATH = os.path.join(RESULTS_DIR, "overall_metrics.json")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    new_entry = {
        "dataset": dataset_name,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss)
    }

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, 'r') as f:
            metrics_log = json.load(f)
    else:
        metrics_log = {}

    if model_name not in metrics_log:
        metrics_log[model_name] = {}

    metrics_log[model_name][timestamp] = new_entry

    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics_log, f, indent=4)

    print(f"📊 Overall metrics saved to {METRICS_PATH}")
