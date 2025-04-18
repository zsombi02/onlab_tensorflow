import os

import tensorflow as tf

from models import BaseModel
from utils.data_utils import dataset_basic_statistics
from utils.train_utils import plot_training_history, save_training_history, save_model_architecture_plot

RESULTS_DIR = "../results/"
MODEL_DIR = "../models/saved_models/"
ARCH_DIR = os.path.join(RESULTS_DIR, "architecture")
CM_DIR = os.path.join(RESULTS_DIR, "confusion_matrixes")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(ARCH_DIR, exist_ok=True)
os.makedirs(CM_DIR, exist_ok=True)

class TrainingPipeline:
    def __init__(self, model_cls: type[BaseModel], model_name="cnn_model", epochs=10, dataset_loader=None, callbacks=None):
        if callbacks is None:
            callbacks = []
        self.model_name = model_name
        self.epochs = epochs
        self.model_cls = model_cls
        self.dataset_loader = dataset_loader or load_cifar10  # Default to CIFAR-10 if not specified
        self.model = None
        self.train_ds, self.test_ds = None, None
        self.callbacks = callbacks

    def load_data(self):
        print(f"📦 Loading dataset using: {self.dataset_loader.__name__}...")
        self.train_ds, self.test_ds = self.dataset_loader()
        dataset_basic_statistics(self.train_ds)
        # plot_random_images(self.train_ds)

    def build_model(self):
        print(f"🧠 Building model: {self.model_name}")
        model_builder: BaseModel = self.model_cls()
        self.model = model_builder.build()
        save_model_architecture_plot(self.model, self.model_name)

    def train(self):
        print("🚀 Starting training...")

        history = self.model.fit(self.train_ds, validation_data=self.test_ds, epochs=self.epochs, callbacks=self.callbacks)

        save_training_history(history, self)
        plot_training_history(history)

    def save_model(self):
        path = os.path.join(MODEL_DIR, f"{self.model_name}.keras")
        self.model.save(path)
        print(f"💾 Model saved at {path}")

    def run(self):
        self.load_data()
        self.build_model()
        self.train()
        self.save_model()
        #self.evaluate_and_plot_confusion_matrix()


if __name__ == "__main__":
    from models.simple_cnn_dropout import SimpleCNNModel
    from data.cifar10 import load_cifar10, load_cifar10_halved, load_cifar10_quartered, load_cifar10_eight, \
    load_cifar10_quartered_fewer_frogs, load_cifar10_quartered_boost_critical_double, \
    load_cifar10_quartered_augmented_double, load_cifar10_quartered_fewer_frogs_augmented_double

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    )

    learning_rate = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6,
                verbose=1
    )

    pipeline = TrainingPipeline(
        model_cls=SimpleCNNModel,
        model_name="Cnn_v4_Quarter_fewer_frogs_augmented_double",
        epochs=75,
        dataset_loader=load_cifar10_quartered_fewer_frogs_augmented_double,
        callbacks=[learning_rate, early_stopping]
    )
    pipeline.run()

