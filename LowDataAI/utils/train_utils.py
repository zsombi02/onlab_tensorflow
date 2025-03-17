import tensorflow as tf
import matplotlib.pyplot as plt
import json
import os
import datetime


def train_model(model, train_ds, val_ds, epochs=10):
    """
    Trains the model with given datasets.

    Args:
        model (tf.keras.Model): The model to train.
        train_ds (tf.data.Dataset): Training dataset.
        val_ds (tf.data.Dataset): Validation dataset.
        epochs (int): Number of training epochs.

    Returns:
        history: The training history object.
    """
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    return history

def plot_training_history(history):
    """
    Plots training and validation accuracy and loss.

    Args:
        history: Training history returned by model.fit.
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


HISTORY_DIR = "../results/history/"
os.makedirs(HISTORY_DIR, exist_ok=True)

def save_training_history(history, model_name):
    """
    Saves training history into a timestamped JSON file inside the history directory.

    Args:
        history: The training history object returned by model.fit().
        model_name (str): Name of the model to save history for.
    """
    history_file = os.path.join(HISTORY_DIR, f"{model_name}_history.json")

    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Load existing history if available
    if os.path.exists(history_file):
        with open(history_file, 'r') as f:
            existing_history = json.load(f)
    else:
        existing_history = {}

    # Append new training history with timestamp
    existing_history[timestamp] = history.history

    # Save updated history file
    with open(history_file, 'w') as f:
        json.dump(existing_history, f, indent=4)

    print(f"Training history saved to {history_file}")