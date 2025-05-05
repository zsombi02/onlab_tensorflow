import os
import tensorflow as tf
from models.BaseModel import BaseModel
from utils.data_utils import ensure_model_has_input


class CombinedCNNModel(BaseModel):
    def __init__(self,
                 animal_model_filename="Quartered_Animals_V2.keras",
                 non_animal_model_filename="Quartered_Non_Animals_V2.keras",
                 input_shape=(32, 32, 3),
                 num_classes=10):
        self.animal_model_filename = animal_model_filename
        self.non_animal_model_filename = non_animal_model_filename
        self.input_shape = input_shape
        self.num_classes = num_classes

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.animal_model_path = os.path.join(current_dir, "saved_models", self.animal_model_filename)
        self.non_animal_model_path = os.path.join(current_dir, "saved_models", self.non_animal_model_filename)

    def build(self) -> tf.keras.Model:
        print(f"📥 Loading animal model from: {self.animal_model_path}")
        animal_model = ensure_model_has_input(tf.keras.models.load_model(self.animal_model_path), self.input_shape)
        animal_model._name = "animal_model"

        print(f"📥 Loading non-animal model from: {self.non_animal_model_path}")
        non_animal_model = ensure_model_has_input(tf.keras.models.load_model(self.non_animal_model_path),
                                                  self.input_shape)
        non_animal_model._name = "non_animal_model"

        animal_model.trainable = False
        non_animal_model.trainable = False

        shared_input = tf.keras.Input(shape=self.input_shape, name="combined_input")

        animal_output = animal_model(shared_input)
        non_animal_output = non_animal_model(shared_input)

        merged = tf.keras.layers.Concatenate()([animal_output, non_animal_output])
        x = tf.keras.layers.Dense(128, activation='relu')(merged)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)

        model = tf.keras.Model(inputs=shared_input, outputs=outputs, name="CombinedCNNModel")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        model.summary()
        return model



