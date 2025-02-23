from models.model_beans import build_cnn_model
from scripts.load_beans import load_dataset

def train_model():

    train_ds, val_ds, test_ds = load_dataset()

    model = build_cnn_model()

    print("Starting training...")
    model.fit(train_ds, validation_data=val_ds, epochs=5)

    # Save model after training
    model.save("models/beans_classifier.h5")
    print("Training complete! Model saved.")

if __name__ == "__main__":
    train_model()
