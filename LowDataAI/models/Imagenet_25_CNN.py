import tensorflow as tf
from models.BaseModel import BaseModel
from tensorflow.keras import regularizers

L = tf.keras.layers


def cosine_lr(init_lr=1e-3, decay_steps=3000, alpha=0.1):
    return tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=init_lr,
        decay_steps=decay_steps,
        alpha=alpha
    )


class SimpleImagenetCNN_25(BaseModel):
    def __init__(self, input_shape=(96, 96, 3), num_classes=25, model_name="simple_imagenet_cnn_v6"):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_name = model_name
        self.reg = 1e-4
        self.head_dropout = 0.50
        self.block_dropout = 0.35
        self.init_lr = 3e-4
        self.wd = 1e-4
        self.decay_steps = 2500

    def build(self) -> tf.keras.Model:
        inputs = L.Input(shape=self.input_shape, name=f"{self.model_name}_input")
        x = inputs

        def build_sep_block(input_tensor, out_ch, name_prefix):
            # 1. Separable Convolution
            x_b = L.SeparableConv2D(
                out_ch, 3, padding="same",
                depthwise_regularizer=regularizers.l2(self.reg),
                pointwise_regularizer=regularizers.l2(self.reg),
                name=f"{name_prefix}_sep"
            )(input_tensor)

            # 2. Batch Normalization
            x_b = L.BatchNormalization(dtype="float32", name=f"{name_prefix}_bn")(x_b)

            # 3. ReLU
            x_b = L.ReLU(name=f"{name_prefix}_relu")(x_b)

            # 4. Dropout (ha drop > 0)
            x_b = L.Dropout(self.block_dropout, name=f"{name_prefix}_drop")(x_b)

            return x_b


        def build_conv_block(input_tensor, out_ch, name_prefix):
            """
            Egyszerű, hagyományos Conv2D blokk
            (SeparableConv2D helyett, a stabilabb 'from scratch' tanítás érdekében)
            """
            # 1. Standard Convolution
            x_b = L.Conv2D(
                out_ch, 3, padding="same",
                kernel_regularizer=regularizers.l2(self.reg),
                name=f"{name_prefix}_conv"
            )(input_tensor)

            # 2. Batch Normalization
            x_b = L.BatchNormalization(dtype="float32", name=f"{name_prefix}_bn")(x_b)

            # 3. ReLU Aktiváció
            x_b = L.ReLU(name=f"{name_prefix}_relu")(x_b)

            # 4. Dropout
            x_b = L.Dropout(self.block_dropout, name=f"{name_prefix}_drop")(x_b)

            return x_b

        # Stem + blokkok
        x = L.Conv2D(96, 3, padding="same", kernel_regularizer=regularizers.l2(self.reg))(x)
        x = L.BatchNormalization(dtype="float32")(x);
        x = L.ReLU()(x)
        x = L.MaxPooling2D()(x)  # 96->48
        x = L.Dropout(self.block_dropout)(x)
        # 1. Blokk: 48x48
        x = build_conv_block(x, 128, name_prefix="conv_128_a")
        x = L.MaxPooling2D()(x)  # 48x48 -> 24x24

        # 2. Blokk: 24x24
        x = build_conv_block(x, 156, name_prefix="conv_192_a")
        x = L.MaxPooling2D()(x)  # 24x24 -> 12x12

        # 3. Blokk: 12x12
        x = build_conv_block(x, 192, name_prefix="conv_256_a")
        x = L.MaxPooling2D()(x)  # 12x12 -> 6x6

        # 4. Blokk: 6x6 (nincs Pooling utána)
        x = build_conv_block(x, 256, name_prefix="conv_320_a")

        x = L.GlobalAveragePooling2D()(x)
        x = L.Dense(128, activation="swish", kernel_regularizer=regularizers.l2(self.reg))(x)
        x = L.Dropout(self.head_dropout)(x)

        # Softmax kimenet
        outputs = L.Dense(self.num_classes, activation="softmax", name="probs", dtype="float32")(x)

        lr = cosine_lr(init_lr=self.init_lr, decay_steps=self.decay_steps, alpha=0.1)

        model = tf.keras.Model(inputs, outputs, name=self.model_name)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=lr
            ),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
            metrics=["accuracy"],
        )
        return model