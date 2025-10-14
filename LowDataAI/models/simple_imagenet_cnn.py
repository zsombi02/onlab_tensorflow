# models/simple_imagenet_cnn.py
import tensorflow as tf
from models.BaseModel import BaseModel
L = tf.keras.layers

class SimpleImagenetCNN(BaseModel):
    def __init__(self, input_shape=(64,64,3), num_classes=100, dropout_rate=0.1, model_name="simple_imagenet_cnn"):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model_name = model_name

    def build(self) -> tf.keras.Model:
        inputs = L.Input(shape=self.input_shape, name=f"{self.model_name}_input")
        x = inputs

        x = L.Conv2D(64, 3, padding="same")(x);
        x = L.BatchNormalization(dtype="float32")(x);
        x = L.ReLU()(x);
        x = L.MaxPooling2D()(x);
        x = L.Dropout(0.05)(x)  # 0.025 -> 0.05
        x = L.Conv2D(128, 3, padding="same")(x);
        x = L.BatchNormalization(dtype="float32")(x);
        x = L.ReLU()(x);
        x = L.MaxPooling2D()(x);
        x = L.Dropout(0.1)(x)  # 0.05 -> 0.1
        x = L.Conv2D(256, 3, padding="same")(x);
        x = L.BatchNormalization(dtype="float32")(x);
        x = L.ReLU()(x);
        x = L.MaxPooling2D()(x);
        x = L.Dropout(0.1)(x)  # 0.05 -> 0.1
        x = L.Conv2D(512, 3, padding="same")(x);
        x = L.BatchNormalization(dtype="float32")(x);
        x = L.ReLU()(x);
        x = L.MaxPooling2D()(x);
        x = L.Dropout(0.15)(x)  # 0.075 -> 0.15
        x = L.Conv2D(1024, 3, padding="same")(x);
        x = L.BatchNormalization(dtype="float32")(x);
        x = L.ReLU()(x);
        x = L.MaxPooling2D()(x);
        x = L.Dropout(0.2)(x)  # 0.1 -> 0.2

        x = L.GlobalAveragePooling2D()(x)
        x = L.Dense(768, activation="swish")(x)
        x = L.Dense(768, activation="swish")(x)
        x = L.Dense(384, activation="swish")(x)
        x = L.Dense(384, activation="swish")(x)
        x = L.Dropout(self.dropout_rate * 2)(x)  # Dönthetsz itt is emelésről, pl. 0.1 -> 0.2

        # Softmax kimenet + FP32 (BN már FP32)
        outputs = L.Dense(self.num_classes, activation="softmax", name="probs", dtype="float32")(x)

        model = tf.keras.Model(inputs, outputs, name=self.model_name)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )
        model.summary()
        return model
