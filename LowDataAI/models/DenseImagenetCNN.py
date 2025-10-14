# models/DenseImagenetCNN.py
import tensorflow as tf
from models.BaseModel import BaseModel
L = tf.keras.layers

class DenseImagenetCNN(BaseModel):
    def __init__(self, input_shape=(64,64,3), num_classes=100, dropout_rate=0.4, model_name="dense_imagenet_cnn"):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model_name = model_name

    def _conv_block(self, x, filters, drop=0.15):
        # 2x Conv + ReLU, majd MaxPool + Dropout — NINCS BatchNorm
        x = L.Conv2D(filters, 3, padding="same", use_bias=True)(x)
        x = L.ReLU()(x)
        x = L.Conv2D(filters, 3, padding="same", use_bias=True)(x)
        x = L.ReLU()(x)
        x = L.MaxPooling2D()(x)
        x = L.Dropout(drop)(x)
        return x

    def build(self) -> tf.keras.Model:
        inputs = L.Input(shape=self.input_shape, name=f"{self.model_name}_input")
        x = inputs

        # mélyebb feature-extractor (nagyjából 4 blokk)
        x = self._conv_block(x, 64,  drop=0.10)
        x = self._conv_block(x, 128, drop=0.15)
        x = self._conv_block(x, 256, drop=0.20)
        x = self._conv_block(x, 512, drop=0.20)

        # GAP + mély, 5 rétegű dense torony (legalább 5× annyi dense, mint az 1 réteg)
        x = L.GlobalAveragePooling2D()(x)
        x = L.Dense(768, activation="swish")(x); x = L.Dropout(self.dropout_rate)(x)
        x = L.Dense(768, activation="swish")(x); x = L.Dropout(self.dropout_rate)(x)
        x = L.Dense(384,  activation="swish")(x); x = L.Dropout(self.dropout_rate * 0.9)(x)
        x = L.Dense(384,  activation="swish")(x); x = L.Dropout(self.dropout_rate * 0.9)(x)
        x = L.Dense(256,  activation="swish")(x); x = L.Dropout(self.dropout_rate * 0.8)(x)

        # Softmax kimenet FP32-n (mixed precision mellett is stabil)
        outputs = L.Dense(self.num_classes, activation="softmax", name="probs", dtype="float32")(x)

        model = tf.keras.Model(inputs, outputs, name=self.model_name)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4, decay=1e-4),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )
        model.summary()
        return model
