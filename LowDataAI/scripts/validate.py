import os
import json
import tensorflow as tf
from data.cifar10 import load_cifar10

# Create directory for evaluation results
RESULTS_DIR = "../results/"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load test dataset
_, test_ds = load_cifar10()

# Load trained model
model_path = "../models/saved_models/cnn_dropout_v2.keras"
model = tf.keras.models.load_model(model_path)
print(f"Loaded model from {model_path}")

# Evaluate
test_loss, test_acc = model.evaluate(test_ds)

# Store results in a JSON file
results = {"Test Accuracy": test_acc, "Test Loss": test_loss}
results_path = os.path.join(RESULTS_DIR, "evaluation_results.json")

with open(results_path, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Evaluation results saved to {results_path}")
print(f"Test Accuracy: {test_acc * 100:.2f}% | Test Loss: {test_loss:.4f}")
