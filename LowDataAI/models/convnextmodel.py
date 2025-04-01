import tensorflow as tf


from models.BaseModel import BaseModel

class ConvNextExtendedModel(BaseModel):
    def __init__(self, input_shape=(32, 32, 3), num_classes=10, dropout_rate=0.5, train_backbone=False):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.train_backbone = train_backbone

    def build(self) -> tf.keras.Model:
        # Load ConvNeXtSmall backbone
        convnext_base = tf.keras.applications.ConvNeXtSmall(
            include_top=False,
            weights='imagenet',
            input_shape=self.input_shape,
            pooling=None
        )

        # Freeze backbone layers if specified
        convnext_base.trainable = self.train_backbone

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Rescaling(1./255, input_shape=self.input_shape))  # Normalize CIFAR-10 images
        model.add(convnext_base)
        model.add(tf.keras.layers.GlobalAveragePooling2D())
        model.add(tf.keras.layers.Dropout(self.dropout_rate))
        model.add(tf.keras.layers.Dense(512, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(self.dropout_rate))
        model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        model.summary()
        return model
