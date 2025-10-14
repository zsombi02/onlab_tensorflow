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

# class SepBlock(tf.keras.layers.Layer):
#     def __init__(self, out_ch, l2=1e-4, drop=0.0, name=None):
#         super().__init__(name=name)
#         self.out_ch = int(out_ch)
#         self.l2 = float(l2)
#         self.drop = float(drop)
#
#         self.sep = L.SeparableConv2D(
#             self.out_ch, 3, padding="same",
#             depthwise_regularizer=regularizers.l2(self.l2),
#             pointwise_regularizer=regularizers.l2(self.l2)
#         )
#         self.bn  = L.BatchNormalization(dtype="float32")
#         self.act = L.ReLU()
#         self.do  = L.Dropout(self.drop) if self.drop > 0.0 else None
#
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "out_ch": self.out_ch,
#             "l2": self.l2,
#             "drop": self.drop,
#         })
#         return config
#
#     def call(self, x, training=None):
#         x = self.sep(x, training=training)
#         x = self.bn(x, training=training)
#         x = self.act(x)
#         if self.do is not None:
#             x = self.do(x, training=training)
#         return x
#

class SimpleImagenetCNN_V5(BaseModel):
    def __init__(self, input_shape=(96, 96, 3), num_classes=100, model_name="simple_imagenet_cnn_v6"):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_name = model_name
        self.reg = 1e-4
        self.head_dropout = 0.50
        self.block_dropout = 0.35
        self.init_lr = 3e-4
        self.wd = 1e-4
        self.decay_steps = 5000

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

        # Stem + blokkok
        x = L.Conv2D(96, 3, padding="same", kernel_regularizer=regularizers.l2(self.reg))(x)
        x = L.BatchNormalization(dtype="float32")(x);
        x = L.ReLU()(x)
        x = L.MaxPooling2D()(x)  # 96->48
        x = L.Dropout(self.block_dropout)(x)

        # A SepBlock hívások cserélve a segédfüggvény hívásokra:
        x = build_sep_block(x, 128, name_prefix="sep_128_a")
        x = L.MaxPooling2D()(x)
        x = build_sep_block(x, 156, name_prefix="sep_192_a")
        x = L.MaxPooling2D()(x)

        x = build_sep_block(x, 192, name_prefix="sep_256_a")
        x = L.MaxPooling2D()(x)

        x = build_sep_block(x, 256, name_prefix="sep_320_a")

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