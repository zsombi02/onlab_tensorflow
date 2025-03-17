import tensorflow as tf
import keras as ke
from keras import Model
from keras.src.applications.mobilenet_v2 import MobileNetV2
from keras.src.layers import GlobalAveragePooling2D, Dense, Dropout


def build_mobilenet_model(input_shape=(32, 32, 3), num_classes=10, dropout_rate=0.4, model_name="mobilenet_cnn"):
    """
    Builds a MobileNetV2-based model for CIFAR-10 classification.

    Args:
        input_shape (tuple): Shape of input images.
        num_classes (int): Number of output classes.
        dropout_rate (float): Dropout rate for regularization.
        model_name (str): Custom name for the model.

    Returns:
        A compiled TensorFlow MobileNetV2 model.
    """
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights=None)  # No pre-trained weights
    base_model.trainable = True  # Enable fine-tuning

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=True)
    x = GlobalAveragePooling2D()(x)  # Replace Flatten with GAP for better performance
    x = Dense(256, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name=model_name)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    return model
