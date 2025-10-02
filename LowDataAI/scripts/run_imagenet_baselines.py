# runners/run_imagenet_baselines.py
import os

import tensorflow as tf

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
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)

    if dataset == "tiny":
        input_shape = (64, 64, 3);
        num_classes = 200
        base_name = "tiny_imagenet"
    else:
        input_shape = (224, 224, 3);
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
    else:
        def full_loader(**_):
            from data.imagenet_wrappers import imnet100_full
            return imnet100_full(batch_size=bs, image_size=isize)

        # def budget_loader(p):
        #     def _loader(**_):
        #         from data.imagenet_wrappers import imnet100_budget
        #         return imnet100_budget(p, batch_size=bs, image_size=isize, seed=seed)

            # return _loader

    # (Opcionálisan) baseline "full train" futtatás is:
    # full_model_name = f"{base_name}_from_scratch_full"
    # train_and_validate(full_model_name, input_shape, num_classes, full_loader, epochs, [reduce_lr, early_stopping], batch_size)

    # Label-budget futások
    for p in budgets:
        print("Budget:", p)
        tag = f"{int(p*100)}pct"
        model_name = f"{base_name}_from_scratch_{tag}"

        # --- Train ---
        tp = TrainingPipeline(
            model_cls=lambda: SimpleImagenetCNN(input_shape=input_shape, num_classes=num_classes, model_name=model_name),
            model_name=model_name,
            epochs=epochs,
            dataset_loader=budget_loader(p),   # <- (train_sub, val) a loader visszatérési értéke
            callbacks=[reduce_lr, early_stopping]
        )
        tp.run()

        # --- Validate (Top1, Macro-F1, ECE + cm/report mentés) ---
        vp = ValidationPipeline(
            model_name=model_name,
            dataset_loader=full_loader,        # <- ugyanarra a val-ra húzunk be (a full wrapper (train, val)-t ad)
            batch_size=batch_size
        )
        vp.run()

if __name__ == "__main__":
    # Tiny-ImageNet sorozat:
    run_series(dataset="tiny", budgets=(0.10, 0.20, 0.30, 0.50), epochs=75, batch_size=64)
    # run_series(dataset="tiny", budgets=(0.50, 0.6), epochs=50, batch_size=64)

    # Ha szeretnéd az ImageNet-100-at is:
    # run_series(dataset="imnet100", budgets=(0.10, 0.20, 0.30, 0.50), epochs=50, batch_size=64)
