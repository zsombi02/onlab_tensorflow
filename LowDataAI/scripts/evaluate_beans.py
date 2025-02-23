import tensorflow as tf
from scripts.load_beans import load_dataset

def evaluate_model():

    model = tf.keras.models.load_model("models/beans_classifier.h5")
    _, _, test_ds = load_dataset()

    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    evaluate_model()
