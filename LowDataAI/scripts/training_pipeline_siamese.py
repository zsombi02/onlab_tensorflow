import os
import tensorflow as tf
from models.siamese import SiameseModel
from data.cifar10 import load_cifar10_quartered_animals_siamese

os.makedirs("../models/saved_models", exist_ok=True)

def run_siamese_training():
    # 🔹 1. Adat betöltése
    (train_X, train_y), (val_X, val_y) = load_cifar10_quartered_animals_siamese()

    train_ds = tf.data.Dataset.from_tensor_slices(((train_X[0], train_X[1]), train_y)).batch(32).prefetch(
        tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_tensor_slices(((val_X[0], val_X[1]), val_y)).batch(32).prefetch(tf.data.AUTOTUNE)

    # 🔹 2. Modell létrehozása
    model = SiameseModel().build()

    # 🔹 3. Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    # 🔹 4. Tréning
    model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=[early_stopping, reduce_lr])

    # 🔹 5. Mentés
    model.save("../models/saved_models/Quartered_Animals_Siamese.keras")
    print("✅ Model saved.")

if __name__ == "__main__":
    run_siamese_training()
