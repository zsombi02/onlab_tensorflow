import os
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = "../results/"
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
ANALYSIS_DIR = os.path.join(REPORTS_DIR, "analysis_results")

CM_JSON_DIR = os.path.join(REPORTS_DIR, "confusion_matrix_jsons")
CR_JSON_DIR = os.path.join(REPORTS_DIR, "classification_reports_jsons")

ANALYSIS_CLASSWISE_DIR = os.path.join(ANALYSIS_DIR, "classwise_results")
ANALYSIS_OVERALL_DIR = os.path.join(ANALYSIS_DIR, "overall_results")

LABEL_FILE = os.path.join(ANALYSIS_DIR, "label.labels.txt")
METRICS_PATH = os.path.join(REPORTS_DIR, "overall_metrics.json")

def load_classification_reports(json_dir, model_names):
    reports = {}
    for model_name in model_names:
        filename = f"{model_name}_classification_report.json"
        path = os.path.join(json_dir, filename)
        if not os.path.exists(path):
            print(f"❌ Missing file: {filename}")
            continue

        with open(path, "r") as f:
            try:
                data = json.load(f)
                reports[model_name] = data
            except json.JSONDecodeError:
                print(f"⚠️ Could not parse JSON in {filename}")
    return reports


def extract_latest_class_metrics(reports_dict, metric="f1-score"):
    data = {}

    for model_name, report_versions in reports_dict.items():
        if not report_versions:
            continue
        # Use the latest timestamp
        latest_timestamp = sorted(report_versions.keys())[-1]
        report = report_versions[latest_timestamp]

        # Extract metrics for each class (only digits)
        class_metrics = {}
        for cls, values in report.items():
            if cls.isdigit() and metric in values:
                class_metrics[int(cls)] = values[metric]

        data[model_name] = class_metrics

    # Build DataFrame and sort index
    df = pd.DataFrame(data)
    df.index.name = "Class"
    df = df.sort_index()
    return df


def plot_classwise_comparison(models, metric=None, title=None, save_path=None, label_file=LABEL_FILE):

    reports = load_classification_reports(CR_JSON_DIR, models)
    df = extract_latest_class_metrics(reports, metric=metric)

    title = f"{title}_{metric}" or f"Class-wise {metric} comparison"

    # Replace numeric index with labels if label_file is provided
    if label_file and os.path.exists(label_file):
        with open(label_file, 'r') as f:
            labels = [line.strip() for line in f.readlines()]
        label_mapping = {i: labels[i] for i in range(len(labels))}
        df = df.rename(index=label_mapping)
    else:
        if label_file:
            print(f"⚠️ Label file not found at: {label_file}. Using numeric class indices.")

    # Plotting
    ax = df.plot(kind="bar", figsize=(12, 6))
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Class")
    ax.set_ylabel(metric.title())
    ax.set_ylim(0, 1)
    ax.grid(True, axis="y")
    plt.xticks(rotation=0, ha="right")
    plt.legend(title="Model")
    plt.tight_layout()

    # Saving
    if save_path:
        if os.path.isdir(save_path):
            import re
            safe_title = re.sub(r"[^\w\-_. ]", "", title).replace(" ", "_")
            save_path = os.path.join(save_path, f"{safe_title}.png")
        plt.savefig(save_path)
        print(f"📊 Plot saved to: {save_path}")
        plt.close()
    else:
        plt.show()


def plot_accuracy_and_loss_from_metrics(metrics_path=METRICS_PATH, save_path=None, title="Test Accuracy & Loss per Model", model_group=None):

    if not os.path.exists(metrics_path):
        print(f"❌ Metrics file not found: {metrics_path}")
        return

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    models = []
    accuracies = []
    losses = []

    for model_name, timestamps in metrics.items():
        if model_group and model_name not in model_group:
            continue
        latest_ts = sorted(timestamps.keys())[-1]
        entry = timestamps[latest_ts]
        models.append(model_name)
        accuracies.append(entry["test_accuracy"])
        losses.append(entry["test_loss"])

    if not models:
        print("⚠️ No models matched the filter. Nothing to plot.")
        return

    x = np.arange(len(models))  # numeric x-ticks

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(x, accuracies, label="Test Accuracy", marker='o', linewidth=2, color='tab:blue')
    ax.plot(x, losses, label="Test Loss", marker='o', linewidth=2, color='tab:red')

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Model")
    ax.set_ylabel("Value")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylim(0, max(max(accuracies), max(losses)) * 1.1)
    ax.legend()
    ax.grid(True, axis='y')
    fig.tight_layout()

    if save_path:
        save_path = os.path.join(ANALYSIS_OVERALL_DIR, f"{title}.png")
        plt.savefig(save_path)
        print(f"📊 Accuracy & loss plot saved to: {save_path}")
        plt.close()
    else:
        plt.show()
