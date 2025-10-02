import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

from data.imagenet_wrappers import tiny_full
from utils.data_utils import dataset_basic_statistics
from utils.validation_utils import log_basic_evaluation_results, save_confusion_matrix_json, \
    save_classification_report_json, plot_confusion_matrix, plot_classification_report, save_overall_metrics

MODEL_DIR = "../models/saved_models/"

class ValidationPipeline:
    def __init__(self, model_name: str, dataset_loader, batch_size: int = 32):
        self.model_name = model_name
        self.dataset_loader = dataset_loader
        self.batch_size = batch_size
        self.model = None
        self.test_ds = None

    def load_model(self):
        path = os.path.join(MODEL_DIR, f"{self.model_name}.keras")
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Model file not found at: {path}")
        self.model = tf.keras.models.load_model(path)
        print(f"✅ Loaded model from: {path}")

    def load_data(self):
        print(f"📦 Loading dataset using: {self.dataset_loader.__name__}...")
        _, self.test_ds = self.dataset_loader(batch_size=self.batch_size)
        dataset_basic_statistics(self.test_ds)

    def evaluate(self):
        test_loss, test_acc = self.model.evaluate(self.test_ds, verbose=2)
        log_basic_evaluation_results(test_acc, test_loss, self.model_name)
        save_overall_metrics(test_acc, test_loss, self.model_name, dataset_name=self.dataset_loader.__name__)

    def evaluate_detailed(self):
        print("📐 Generating predictions and computing metrics...")
        y_true, y_pred = [], []

        for images, labels in self.test_ds:
            preds = self.model.predict(images, verbose=0)
            y_true.extend(labels.numpy())
            y_pred.extend(np.argmax(preds, axis=1))

        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(y_true, y_pred, self.model_name)
        save_confusion_matrix_json(cm, self.model_name)

        report_dict = classification_report(y_true, y_pred, digits=4, output_dict=True)
        report_str = classification_report(y_true, y_pred, digits=4, output_dict=False)

        plot_classification_report(report_str, self.model_name)
        save_classification_report_json(report_dict, self.model_name)

        # ➕ top1, macro-F1, ECE
        all_probs = []
        all_labels = []
        for images, labels in self.test_ds:
            probs = self.model.predict(images, verbose=0)
            all_probs.append(probs)
            all_labels.append(labels.numpy())
        probs = np.concatenate(all_probs, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        from sklearn.metrics import f1_score
        from utils.imagenet_utils import compute_ece

        top1 = (probs.argmax(axis=1) == labels).mean()
        macro_f1 = f1_score(labels, probs.argmax(axis=1), average="macro")
        ece = compute_ece(probs, labels, n_bins=15)

        save_overall_metrics(
            top1,  # accuracy
            0.0,  # nincs loss itt kéznél, maradhat 0 vagy hagyd ki
            self.model_name,
            dataset_name=self.dataset_loader.__name__,
            extra={"macro_f1": float(macro_f1), "ece": float(ece)}
        )

    def run(self):
        self.load_model()
        self.load_data()
        self.evaluate()
        self.evaluate_detailed()



if __name__ == "__main__":

    validator = ValidationPipeline(
        model_name="tiny_imagenet_from_scratch_10pct",
        dataset_loader=tiny_full,
        batch_size=32
    )
    validator.run()
#
#
#     # for i in range(1, 21):  # 1-től 25-ig
#     #     model_name = f"Multirun_Quartered_Animals_V2_Run_{i:02d}"
#     #     print(f"\n🚀 Starting run {i}/25: {model_name}")
#     #
#     #     validator = ValidationPipeline(
#     #         model_name=model_name,
#     #         dataset_loader=load_cifar10_quartered_animals,
#     #         batch_size=32
#     #     )
#     #     validator.run()
#
#     # for i in range(1, 21):  # 1-től 25-ig
#     #     model_name = f"Combined_TopAnimal_{i:02d}"
#     #     print(f"\n🚀 Starting run {i}/25: {model_name}")
#     #
#     #     validator = ValidationPipeline(
#     #         model_name=model_name,
#     #         dataset_loader=load_cifar10_quartered,
#     #         batch_size=32
#     #     )
#     #     validator.run()
#
# #augmentalas egysegesen