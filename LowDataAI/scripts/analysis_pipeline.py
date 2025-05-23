import os
from enum import Enum

from utils.analysis_utils import plot_classwise_comparison, plot_accuracy_and_loss_from_metrics
from utils.model_groups import modelgroup_normal, modelgroup_quartered_augmented, modelgroup_quartered_changed, \
    modelgroup_multirun, modelgroup_combined, modelgroup_combined_top_animals

RESULTS_DIR = "../results/"
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")
ANALYSIS_DIR = os.path.join(REPORTS_DIR, "analysis_results")

CM_JSON_DIR = os.path.join(REPORTS_DIR, "confusion_matrix_jsons")
CR_JSON_DIR = os.path.join(REPORTS_DIR, "classification_reports_jsons")

ANALYSIS_CLASSWISE_DIR = os.path.join(ANALYSIS_DIR, "classwise_results")
ANALYSIS_OVERALL_DIR = os.path.join(ANALYSIS_DIR, "overall_results")

class Metric(str, Enum):
    F1 = "f1-score"
    PRECISION = "precision"



def run_classwise_comparison():

    plot_classwise_comparison(models=modelgroup_combined_top_animals, metric=Metric.F1.value, title="combined_top_animals", save_path=ANALYSIS_CLASSWISE_DIR)


def run_overall_comparison():

    plot_accuracy_and_loss_from_metrics(save_path=ANALYSIS_OVERALL_DIR, title="combined_top_animals", model_group=modelgroup_combined_top_animals)


if __name__ == "__main__":

    run_classwise_comparison()

    run_overall_comparison()