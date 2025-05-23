import os

import tensorflow as tf

from models import BaseModel
from models.combined_model import CombinedCNNModel
from models.siamese import SiameseModel
from models.simpleCnn_named import SimpleCNNModelNamed
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
    load_cifar10_quartered_augmented_double, load_cifar10_quartered_fewer_frogs_augmented_double, \
    load_cifar10_quartered_animals, load_cifar10_quartered_nonanimals, \
    load_cifar10_quartered_animals_save, load_cifar10_halved_animals, load_cifar10_full_animals

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

    # pipeline = TrainingPipeline(
    #     model_cls=SimpleCNNModel,
    #     model_name="Cnn_v4_Quarter_fewer_frogs_augmented_double",
    #     epochs=75,
    #     dataset_loader=load_cifar10_quartered_fewer_frogs_augmented_double,
    #     callbacks=[learning_rate, early_stopping]
    # )
    # pipeline.run()
    #
    # pipeline = TrainingPipeline(
    #     model_cls=SimpleCNNModel,
    #     model_name="Quartered_Animals",
    #     epochs=75,
    #     dataset_loader=load_cifar10_quartered_animals,
    #     callbacks=[learning_rate, early_stopping]
    # )
    # pipeline.run()
    #



    # Ezek jók voltak mostanra!!!

    # pipeline = TrainingPipeline(
    #     model_cls= lambda:  SimpleCNNModelNamed(num_classes=4),
    #     model_name="Quartered_Non_Animals",
    #     epochs=75,
    #     dataset_loader=load_cifar10_quartered_nonanimals,
    #     callbacks=[learning_rate, early_stopping]
    # )
    # pipeline.run()
    #
    # pipeline = TrainingPipeline(
    #     model_cls= lambda:  SimpleCNNModelNamed(num_classes=6),
    #     model_name="Quartered_Animals_V2",
    #     epochs=75,
    #     dataset_loader=load_cifar10_quartered_animals,
    #     callbacks=[learning_rate, early_stopping]
    # )
    # pipeline.run()
    #
    # pipeline = TrainingPipeline(
    #     model_cls= CombinedCNNModel,
    #     model_name="Quartered_Combined_V2",
    #     epochs=75,
    #     dataset_loader=load_cifar10_quartered,
    #     callbacks=[learning_rate, early_stopping]
    # )
    #
    # pipeline.run()


    # Ezek jók voltak mostanra!!!

    # pipeline = TrainingPipeline(
    #     model_cls= lambda:  SimpleCNNModelNamed(num_classes=4),
    #     model_name="Quartered_Non_Animals",
    #     epochs=75,
    #     dataset_loader=load_cifar10_quartered_nonanimals,
    #     callbacks=[learning_rate, early_stopping]
    # )
    # pipeline.run()
    #
    # pipeline = TrainingPipeline(
    #     model_cls= lambda:  SimpleCNNModelNamed(num_classes=6),
    #     model_name="Quartered_Animals_V2",
    #     epochs=75,
    #     dataset_loader=load_cifar10_quartered_animals,
    #     callbacks=[learning_rate, early_stopping]
    # )
    # pipeline.run()
    #
    # pipeline = TrainingPipeline(
    #     model_cls= CombinedCNNModel,
    #     model_name="Quartered_Combined_V2",
    #     epochs=75,
    #     dataset_loader=load_cifar10_quartered,
    #     callbacks=[learning_rate, early_stopping]
    # )

    # for i in range(1, 26):  # 1-től 25-ig
    #     model_name = f"Multirun_Quartered_Animals_V2_Run_{i:02d}"
    #     print(f"\n🚀 Starting run {i}/25: {model_name}")
    #
    #     pipeline = TrainingPipeline(
    #         model_cls=lambda: SimpleCNNModelNamed(num_classes=6),
    #         model_name=model_name,
    #         epochs=75,
    #         dataset_loader=load_cifar10_quartered_animals_save,
    #         callbacks=[learning_rate, early_stopping]
    #     )
    #     pipeline.run()

    # pipeline = TrainingPipeline(
    #     model_cls= lambda:  SimpleCNNModelNamed(num_classes=6),
    #     model_name="Halved_Animals_V2",
    #     epochs=75,
    #     dataset_loader=load_cifar10_halved_animals,
    #     callbacks=[learning_rate, early_stopping]
    # )
    # pipeline.run()
    #
    # pipeline = TrainingPipeline(
    #     model_cls= lambda:  SimpleCNNModelNamed(num_classes=6),
    #     model_name="Full_Animals_V2",
    #     epochs=75,
    #     dataset_loader=load_cifar10_full_animals,
    #     callbacks=[learning_rate, early_stopping]
    # )
    # pipeline.run()
    #
    # top_5_animal_models = [
    #     "Multirun_Quartered_Animals_V2_Run_06.keras",
    #     "Multirun_Quartered_Animals_V2_Run_05.keras",
    #     "Multirun_Quartered_Animals_V2_Run_20.keras",
    #     "Multirun_Quartered_Animals_V2_Run_15.keras",
    #     "Multirun_Quartered_Animals_V2_Run_16.keras"
    # ]
    #
    # for i, animal_model in enumerate(top_5_animal_models, start=1):
    #     model_name = f"Combined_TopAnimal_{i:02d}"
    #     print(f"\n🚀 Starting Combined Model Run {i}: {model_name} using {animal_model}")
    #
    pipeline = TrainingPipeline(
            model_cls=lambda: CombinedCNNModel(
                animal_model_filename="Full_Animals_V2.keras",
                non_animal_model_filename="Quartered_Non_Animals_V2.keras",
                input_shape=(32, 32, 3),
                num_classes=10
            ),
            model_name="Combined_FullAnimal_QuarterNonAnimal",
            epochs=75,
            dataset_loader=load_cifar10_quartered,
            callbacks=[learning_rate, early_stopping]
    )
    pipeline.run()
