import tensorflow as tf
import matplotlib.pyplot as plt
import json
import os

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

def save_training_history(history, filepath):
    """
    Saves training history to a JSON file.

    Args:
        history: The history object returned by model.fit.
        filepath (str): Path to save the history JSON.
    """
    history_dict = history.history
    with open(filepath, 'w') as f:
        json.dump(history_dict, f)
    print(f"Training history saved to {filepath}")

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
