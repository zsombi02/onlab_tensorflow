import os

from keras.src.utils import plot_model
from win32api import Sleep

from data.cifar10 import load_cifar10
from models.mobilenet import build_mobilenet_model
from models.simple_cnn_dropout import build_cnn_model
from utils.statistics_utils import dataset_basic_statistics
from utils.train_utils import train_model, plot_training_history, save_training_history
from utils.visualization_utils import plot_random_images

# Create directories for saved models if they don't exist
MODEL_DIR = "../models/saved_models/"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
train_ds, test_ds = load_cifar10()

#Display dataset statistics
dataset_basic_statistics(train_ds)
plot_random_images(train_ds)
Sleep(5000)

# Specify model name
model_name = "mobilenet_v2_cnn"

# Build model with custom name
model = build_mobilenet_model()

# Plot model
plot_model(model, to_file=f"../results/architecture/{model_name}_architecture.png", show_shapes=True, show_layer_names=True)
print(f"Model architecture saved as '../results/architecture/{model_name}_architecture.png'")


# Train model
trained_model = train_model(model, train_ds, test_ds, epochs=10)

# Save training history using the function from train_utils.py
save_training_history(trained_model, model_name)

# Plot training history
plot_training_history(trained_model)

# Save trained model
model_path = os.path.join(MODEL_DIR, f"{model_name}.keras")
model.save(model_path)
print(f"Model saved at {model_path}")
