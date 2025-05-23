import tensorflow as tf
from models.BaseModel import BaseModel


def contrastive_loss(margin=1.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(margin - y_pred, 0))
        return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)
    return loss


class SiameseModel(BaseModel):
    def __init__(self, input_shape=(32, 32, 3), dropout_rate=0.2, model_name="siamese"):
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate
        self.model_name = model_name

    def build_base_network(self) -> tf.keras.Model:
        model = tf.keras.Sequential(name=f"{self.model_name}_base")

        model.add(tf.keras.layers.InputLayer(input_shape=self.input_shape, name=f"{self.model_name}_input"))

        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', name=f"{self.model_name}_conv1"))
        model.add(tf.keras.layers.BatchNormalization(name=f"{self.model_name}_bn1"))
        model.add(tf.keras.layers.LeakyReLU(name=f"{self.model_name}_lrelu1"))
        model.add(tf.keras.layers.MaxPooling2D((2, 2), name=f"{self.model_name}_pool1"))
        model.add(tf.keras.layers.Dropout(0.2, name=f"{self.model_name}_dropout1"))

        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name=f"{self.model_name}_conv2"))
        model.add(tf.keras.layers.BatchNormalization(name=f"{self.model_name}_bn2"))
        model.add(tf.keras.layers.MaxPooling2D((2, 2), name=f"{self.model_name}_pool2"))
        model.add(tf.keras.layers.Dropout(0.25, name=f"{self.model_name}_dropout2"))

        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name=f"{self.model_name}_conv3"))
        model.add(tf.keras.layers.BatchNormalization(name=f"{self.model_name}_bn3"))
        model.add(tf.keras.layers.GlobalAveragePooling2D(name=f"{self.model_name}_gap"))

        model.add(tf.keras.layers.Dense(256, activation='relu', name=f"{self.model_name}_dense1"))
        model.add(tf.keras.layers.BatchNormalization(name=f"{self.model_name}_bn4"))
        model.add(tf.keras.layers.Dropout(self.dropout_rate, name=f"{self.model_name}_dropout3"))

        model.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1), name=f"{self.model_name}_l2norm"))

        return model

    def build(self) -> tf.keras.Model:
        base_network = self.build_base_network()

        input_a = tf.keras.Input(shape=self.input_shape, name=f"{self.model_name}_input_a")
        input_b = tf.keras.Input(shape=self.input_shape, name=f"{self.model_name}_input_b")

        emb_a = base_network(input_a)
        emb_b = base_network(input_b)

        # Euclidean distance
        distance = tf.keras.layers.Lambda(
            lambda tensors: tf.sqrt(tf.reduce_sum(tf.square(tensors[0] - tensors[1]), axis=1, keepdims=True)),
            name=f"{self.model_name}_distance"
        )([emb_a, emb_b])

        model = tf.keras.Model(inputs=[input_a, input_b], outputs=distance, name=f"{self.model_name}_model")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
            loss=contrastive_loss(margin=0.5),
            metrics=['accuracy']
        )

        model.summary()
        return model
