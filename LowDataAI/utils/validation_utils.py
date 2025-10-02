import os

import os
import json
import datetime
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score

from utils.imagenet_utils import compute_ece

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



def plot_confusion_matrix(y_true, y_pred, model_name):
    # cm = confusion_matrix(y_true, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    # disp.plot(cmap="Blues", xticks_rotation="vertical")
    # plt.title(f"Confusion Matrix - {model_name}")
    #
    cm_path = os.path.join(CM_DIR, f"{model_name}_confusion_matrix.png")
    # plt.savefig(cm_path)
    # plt.show()
    #
    print(f"🖼️ Confusion matrix saved to {cm_path}")
#     TODO túl nagy most kiplottolni, JSONben elég

def _get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def save_classification_report_json(report_dict, model_name):
    timestamp = _get_timestamp()
    path = os.path.join(CR_JSON_DIR, f"{model_name}_classification_report.json")

    # ⬇️ itt a fontos rész
    report_dict = _json_sanitize_tree(report_dict)

    if os.path.exists(path):
        with open(path, 'r', encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = {}

    existing[timestamp] = report_dict

    with open(path, 'w', encoding="utf-8") as f:
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

def save_overall_metrics(test_acc, test_loss, model_name, dataset_name="default", extra=None):
    RESULTS_DIR = "../results/reports/"
    METRICS_PATH = os.path.join(RESULTS_DIR, "overall_metrics.json")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    new_entry = {
        "dataset": dataset_name,
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss)
    }
    if extra:
        new_entry.update(extra)

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



def eval_probs_and_metrics(model, ds):
    all_probs, all_labels = [], []
    for images, labels in ds:
        probs = model.predict(images, verbose=0)
        all_probs.append(probs)
        all_labels.append(labels.numpy())
    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    y_pred = probs.argmax(axis=1)
    top1_acc = (y_pred == labels).mean()
    macro_f1 = f1_score(labels, y_pred, average="macro")
    ece = compute_ece(probs, labels, n_bins=15)
    return top1_acc, macro_f1, ece


def _json_sanitize(x):
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, tf.Tensor):
        x = x.numpy()
        return x.tolist() if hasattr(x, "tolist") else x
    if isinstance(x, set):
        return list(x)
    return x

def _json_sanitize_tree(obj):
    if isinstance(obj, dict):
        return {k: _json_sanitize_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize_tree(v) for v in obj]
    return _json_sanitize(obj)