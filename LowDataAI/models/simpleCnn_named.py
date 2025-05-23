import tensorflow as tf
from models.BaseModel import BaseModel


class SimpleCNNModelNamed(BaseModel):
    def __init__(self, input_shape=(32, 32, 3), num_classes=10, dropout_rate=0.5, model_name="simple_cnn"):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.model_name = model_name

    def build(self) -> tf.keras.Model:
        model = tf.keras.Sequential(name=self.model_name)

        model.add(tf.keras.layers.InputLayer(input_shape=self.input_shape, name=f"{self.model_name}_input"))

        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', name=f"{self.model_name}_conv1"))
        model.add(tf.keras.layers.BatchNormalization(name=f"{self.model_name}_bn1"))
        model.add(tf.keras.layers.LeakyReLU(name=f"{self.model_name}_lrelu1"))
        model.add(tf.keras.layers.MaxPooling2D((2, 2), name=f"{self.model_name}_pool1"))
        model.add(tf.keras.layers.Dropout(0.25, name=f"{self.model_name}_dropout1"))

        model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name=f"{self.model_name}_conv2"))
        model.add(tf.keras.layers.BatchNormalization(name=f"{self.model_name}_bn2"))
        model.add(tf.keras.layers.MaxPooling2D((2, 2), name=f"{self.model_name}_pool2"))
        model.add(tf.keras.layers.Dropout(0.3, name=f"{self.model_name}_dropout2"))

        model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name=f"{self.model_name}_conv3"))
        model.add(tf.keras.layers.BatchNormalization(name=f"{self.model_name}_bn3"))
        model.add(tf.keras.layers.MaxPooling2D((2, 2), name=f"{self.model_name}_pool3"))
        model.add(tf.keras.layers.Dropout(0.35, name=f"{self.model_name}_dropout3"))

        model.add(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name=f"{self.model_name}_conv4"))
        model.add(tf.keras.layers.BatchNormalization(name=f"{self.model_name}_bn4"))
        model.add(tf.keras.layers.MaxPooling2D((2, 2), name=f"{self.model_name}_pool4"))
        model.add(tf.keras.layers.Dropout(0.4, name=f"{self.model_name}_dropout4"))

        model.add(tf.keras.layers.GlobalAveragePooling2D(name=f"{self.model_name}_gap"))

        model.add(tf.keras.layers.Dense(1024, activation='swish', name=f"{self.model_name}_dense1"))
        model.add(tf.keras.layers.BatchNormalization(name=f"{self.model_name}_bn5"))
        model.add(tf.keras.layers.Dropout(self.dropout_rate, name=f"{self.model_name}_dropout5"))

        model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax', name=f"{self.model_name}_output"))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        model.summary()
        return model


# minden dataset méretre kipróbálni csoportokra, és sok random futás animalre