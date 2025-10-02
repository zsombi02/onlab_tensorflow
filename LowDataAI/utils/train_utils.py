import shutil

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import json
import os
import datetime

from tensorflow.keras.utils import plot_model

HISTORY_DIR = "../results/history/"
ARCH_DIR = "../results/architecture/"
os.makedirs(ARCH_DIR, exist_ok=True)
os.makedirs(HISTORY_DIR, exist_ok=True)

def train_model(model, train_ds, val_ds, epochs=10, callbacks=None):
    """
    Trains the model with given datasets.

    Args:
        model (tf.keras.Model): The model to train.
        train_ds (tf.data.Dataset): Training dataset.
        val_ds (tf.data.Dataset): Validation dataset.
        epochs (int): Number of training epochs.
        callbacks (list): List of Keras callbacks to use during training.

    Returns:
        history: The training history object.
    """
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=[callbacks] if callbacks else [])
    return history

def plot_training_history(history):
    """
    Plots training and validation accuracy and loss.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Accuracy
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].set_title('Model Accuracy')

    # Plot Loss
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].set_title('Model Loss')

    plt.show()

def save_training_history(history, pipeline):
    model_name = pipeline.model_name
    history_file = os.path.join(HISTORY_DIR, f"{model_name}_history.json")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # history.history → tiszta python
    clean_hist = {k: [_json_sanitize(v) for v in vals] for k, vals in history.history.items()}

    config = {
        "model_name": pipeline.model_name,
        "epochs": int(pipeline.epochs),
        "model_cls": getattr(pipeline.model_cls, "__name__", str(pipeline.model_cls)),
        "dataset_loader": pipeline.dataset_loader.__name__ if pipeline.dataset_loader else None,
        "callbacks": [type(cb).__name__ for cb in (pipeline.callbacks or [])],
    }

    entry = {"training_config": config, "history": clean_hist}

    existing = {}
    if os.path.exists(history_file):
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except json.JSONDecodeError:
            backup = history_file + f".corrupt_{timestamp}.bak"
            shutil.copy2(history_file, backup)
            print(f"⚠️  Corrupt history JSON → backup: {backup}. Új fájl készül.")
            existing = {}

    existing[timestamp] = entry

    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=4)
    print(f"📝 Training history saved to {history_file}")

def save_model_architecture_plot(model, model_name):
    """
    Saves a visual architecture plot of the model to the architecture folder.

    Args:
        model (tf.keras.Model): The Keras model.
        model_name (str): Filename prefix for the saved plot.
    """
    # path = os.path.join(ARCH_DIR, f"{model_name}_architecture.png")
    # plot_model(
    #     model,
    #     to_file=path,
    #     show_shapes=True,
    #     show_layer_names=True
    # )
    # print(f"📐 Model architecture saved to {path}")
    # TODO pydot

def _json_sanitize(x):
    # numpy skálákat → natív python
    if isinstance(x, (np.generic,)):
        return x.item()
    # numpy tömb → list
    if isinstance(x, np.ndarray):
        return x.tolist()
    # tf.Tensor → list (vagy skála esetén érték)
    if isinstance(x, tf.Tensor):
        x = x.numpy()
        return x.tolist() if hasattr(x, "tolist") else x
    # set → list
    if isinstance(x, set):
        return list(x)
    return x

def _json_sanitize_tree(obj):
    if isinstance(obj, dict):
        return {k: _json_sanitize_tree(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize_tree(v) for v in obj]
    return _json_sanitize(obj)