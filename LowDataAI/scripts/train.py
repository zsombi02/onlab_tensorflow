import os
from data.cifar10 import load_cifar10
from models.simple_cnn import build_cnn_model
from utils.train_utils import train_model, plot_training_history, save_training_history

# Create a directory for saved models if it doesn't exist
MODEL_DIR = "../models/saved_models/"
os.makedirs(MODEL_DIR, exist_ok=True)

# Load dataset
train_ds, test_ds = load_cifar10()

# Build model
model = build_cnn_model()

# Train model
history = train_model(model, train_ds, test_ds, epochs=10)

# Save training history for later analysis
save_training_history(history, os.path.join(MODEL_DIR, "training_history.json"))

# Plot training history
plot_training_history(history)

# Save model
model_path = os.path.join(MODEL_DIR, "trained_model.h5")
model.save(model_path)
print(f"Model saved at {model_path}")
