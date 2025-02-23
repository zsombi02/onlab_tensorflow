import tensorflow as tf

# Check GPU availability
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# Simple TensorFlow test
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[5, 6], [7, 8]])
c = tf.matmul(a, b)

print("Matrix multiplication result:\n", c.numpy())
