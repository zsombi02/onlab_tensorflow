import tensorflow as tf
print("TF:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))

m = tf.keras.Sequential([
    tf.keras.layers.Input((64,64,3)),
    tf.keras.layers.Conv2D(8, 3, activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(200, activation='softmax', dtype='float32'),
])
m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
import numpy as np
x = np.random.rand(64,64,64,3).astype('float32')
y = np.random.randint(0,200,(64,), dtype='int32')
m.fit(x,y, epochs=1, batch_size=16)
