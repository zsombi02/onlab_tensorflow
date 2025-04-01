import os
import json
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from utils.data_utils import dataset_basic_statistics
from utils.validation_utils import plot_classification_report, plot_confusion_matrix, \
    log_basic_evaluation_results

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

    def evaluate_detailed(self):
        print("📐 Generating predictions and computing metrics...")
        y_true = []
        y_pred = []

        for images, labels in self.test_ds:
            preds = self.model.predict(images, verbose=0)
            y_true.extend(labels.numpy())
            y_pred.extend(np.argmax(preds, axis=1))

        # Confusion Matrix
        plot_confusion_matrix(y_true, y_pred, self.model_name)

        # Classification Report
        report = classification_report(y_true, y_pred, digits=4, output_dict=True)
        plot_classification_report(report, self.model_name)



    def run(self):
        self.load_model()
        self.load_data()
        self.evaluate()
        self.evaluate_detailed()



if __name__ == "__main__":
    from data.cifar10 import load_cifar10

    validator = ValidationPipeline(
        model_name="convNextV1",
        dataset_loader=load_cifar10,
        batch_size=32
    )
    validator.run()
