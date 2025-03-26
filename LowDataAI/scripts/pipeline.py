import os
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.src.utils import plot_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from data.cifar10 import load_cifar10
from models import BaseModel
from utils.train_utils import train_model, plot_training_history, save_training_history
from utils.statistics_utils import dataset_basic_statistics
from utils.visualization_utils import plot_random_images

RESULTS_DIR = "../results/"
MODEL_DIR = "../models/saved_models/"
ARCH_DIR = os.path.join(RESULTS_DIR, "architecture")
CM_DIR = os.path.join(RESULTS_DIR, "confusion_matrixes")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ARCH_DIR, exist_ok=True)
os.makedirs(CM_DIR, exist_ok=True)

class TrainingPipeline:
    def __init__(self, model_cls: type[BaseModel], model_name="cnn_model", epochs=30):
        self.model_name = model_name
        self.epochs = epochs
        self.model_cls = model_cls
        self.model = None
        self.train_ds, self.test_ds = None, None

    def load_data(self):
        print("📦 Loading CIFAR-10 dataset...")
        self.train_ds, self.test_ds = load_cifar10()
        dataset_basic_statistics(self.train_ds)
        #plot_random_images(self.train_ds)
        #TODO dataset parameter

    def build_model(self):
        print(f"🧠 Building model: {self.model_name}")
        model_builder: BaseModel = self.model_cls()
        self.model = model_builder.build()
        plot_model(
            self.model,
            to_file=os.path.join(ARCH_DIR, f"{self.model_name}_architecture.png"),
            show_shapes=True,
            show_layer_names=True
        )
        print(f"📐 Model plot saved to {ARCH_DIR}/{self.model_name}_architecture.png")

    def train(self):
        print("🚀 Starting training...")
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=7,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1
        )

        history = train_model(
            self.model,
            self.train_ds,
            self.test_ds,
            epochs=self.epochs,
            callbacks=[early_stopping]
        )

        save_training_history(history, self.model_name)
        plot_training_history(history)

    def save_model(self):
        path = os.path.join(MODEL_DIR, f"{self.model_name}.keras")
        self.model.save(path)
        print(f"💾 Model saved at {path}")

    def evaluate_and_plot_confusion_matrix(self):
        print("📊 Evaluating and generating confusion matrix...")
        y_true = []
        y_pred = []

        for images, labels in self.test_ds:
            preds = self.model.predict(images)
            y_true.extend(labels.numpy())
            y_pred.extend(np.argmax(preds, axis=1))

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
        plt.title(f"Confusion Matrix - {self.model_name}")
        cm_path = os.path.join(CM_DIR, f"{self.model_name}_confusion_matrix.png")
        plt.savefig(cm_path)
        print(f"🖼️ Confusion matrix saved to {cm_path}")
        plt.show()

    def run(self):
        self.load_data()
        self.build_model()
        self.train()
        self.save_model()
        self.evaluate_and_plot_confusion_matrix()


if __name__ == "__main__":
    from models.simple_cnn_dropout import SimpleCNNModel

    pipeline = TrainingPipeline(
        model_cls=SimpleCNNModel,
        model_name="cnn_dropout_v3",
        epochs=50
    )
    pipeline.run()
