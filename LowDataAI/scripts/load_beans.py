import tensorflow as tf
import tensorflow_datasets as tfds

def load_dataset(batch_size=32):

    dataset, info = tfds.load("beans", as_supervised=True, with_info=True)

    train_ds, val_ds, test_ds = dataset['train'], dataset['validation'], dataset['test']

    def preprocess(image, label):
        image = tf.image.resize(image, (128, 128)) / 255.0  # Resize and scale to [0,1]
        return image, label

    train_ds = train_ds.map(preprocess).shuffle(1000).batch(batch_size)
    val_ds = val_ds.map(preprocess).batch(batch_size)
    test_ds = test_ds.map(preprocess).batch(batch_size)

    return train_ds, val_ds, test_ds
