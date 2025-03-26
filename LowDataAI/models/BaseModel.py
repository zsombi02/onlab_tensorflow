from abc import ABC, abstractmethod
import tensorflow as tf

class BaseModel(ABC):
    @abstractmethod
    def build(self) -> tf.keras.Model:
        """
        Abstract method to build and return a compiled Keras model.
        """
        pass
