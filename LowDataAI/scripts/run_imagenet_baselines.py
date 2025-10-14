# runners/run_imagenet_baselines.py
import os

import tensorflow as tf

from models.DenseImagenetCNN import DenseImagenetCNN
from models.Imagenet_50_CNN import SimpleImagenetCNN_V5
from models.simple_imagenet_cnn import SimpleImagenetCNN
from scripts.training_pipeline import TrainingPipeline
from scripts.validation_pipeline import ValidationPipeline

RESULTS_DIR = "../results/"
MODEL_DIR = "../models/saved_models/"
os.makedirs(MODEL_DIR, exist_ok=True)

def run_series(
    dataset="tiny",                     # "tiny" | "imnet100"
    budgets=(0.10, 0.20, 0.30, 0.50),
    epochs=50,
    batch_size=64,
    seed=42
):
    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min",
        patience=12, min_delta=1e-3, restore_best_weights=True
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", mode="min",
        factor=0.5, patience=2, cooldown=1, min_delta=5e-4, min_lr=5e-6, verbose=1 # Hagyja, hogy az LR csökkenjen, amint a Val Loss stagnál
    )
    if dataset == "tiny":
        input_shape = (64, 64, 3)
        num_classes = 200
        base_name = "tiny_imagenet"

    elif dataset == "imnet100_kaggle":
        # input_shape = (224, 224, 3)
        input_shape = (112, 112, 3)
        num_classes = 100
        base_name = "imagenet100_kaggle"

    elif dataset == "imnet50_kaggle":
        # input_shape = (224, 224, 3)
        input_shape = (96, 96, 3)
        num_classes = 50
        base_name = "imagenet50_kaggle"

    else:
        input_shape = (96, 96, 3);
        num_classes = 100
        base_name = "imagenet100"

    # Zárjuk be a külső értékeket tisztán:
    bs = batch_size
    isize = input_shape[:2]

    if dataset == "tiny":
        # fogadjon kwargs-ot, hogy a ValidationPipeline batch_size paramja se zavarjon
        def full_loader(**_):
            from data.imagenet_wrappers import tiny_full
            return tiny_full(batch_size=bs, image_size=isize)

        def budget_loader(p):
            def _loader(**_):
                from data.imagenet_wrappers import tiny_budget
                return tiny_budget(p, batch_size=bs, image_size=isize, seed=seed)

            return _loader

    elif dataset == "imnet100_kaggle":

        def full_loader(**_):
            from data.imagenet_wrappers import imnet100_kaggle_full
            return imnet100_kaggle_full(batch_size=bs, image_size=isize)

        def budget_loader(p):
            def _loader(**_):
                from data.imagenet_wrappers import imnet100_kaggle_budget
                return imnet100_kaggle_budget(p, batch_size=bs, image_size=isize, seed=seed)

            return _loader

    elif dataset == "imnet50_kaggle":

        def full_loader(**_):
            from data.imagenet_wrappers import imnet50_from_100_budget
            return imnet50_from_100_budget(batch_size=bs, image_size=isize)

        def budget_loader(p):
            def _loader(**_):
                from data.imagenet_wrappers import imnet50_from_100_budget
                return imnet50_from_100_budget(p, batch_size=bs, image_size=isize, seed=seed)

            return _loader

    # Label-budget futások
    for p in budgets:
        print("Budget:", p)
        tag = f"{int(p*100)}pct"
        model_name = f"{base_name}_V1_112px_from_scratch_{tag}"

        # --- Train ---
        tp = TrainingPipeline(
            model_cls=lambda: SimpleImagenetCNN_V5(input_shape=input_shape, num_classes=num_classes, model_name=model_name),
            model_name=model_name,
            epochs=epochs,
            dataset_loader=budget_loader(p),   # <- (train_sub, val) a loader visszatérési értéke
            callbacks=[ early_stopping]
        )
        tp.run()

        # --- Validate (Top1, Macro-F1, ECE + cm/report mentés) ---
        vp = ValidationPipeline(
            model_name=model_name,
            dataset_loader=budget_loader(1.0),        # <- ugyanarra a val-ra húzunk be (a full wrapper (train, val)-t ad)
            batch_size=batch_size
        )
        vp.run()


    # os.system('shutdown -s')

if __name__ == "__main__":
    # Tiny-ImageNet sorozat:
    run_series(dataset="imnet50_kaggle", budgets=(0.50, 1.0), epochs=50, batch_size=48)
    # run_series(dataset="tiny", budgets=(0.50, 0.6), epochs=50, batch_size=64)

