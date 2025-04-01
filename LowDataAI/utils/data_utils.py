import collections
import os
import tensorflow as tf
import tensorflow_datasets as tfds

# Set a custom directory for storing datasets
DATASET_DIR = os.path.join("..", "data", "tfds_data")
os.makedirs(DATASET_DIR, exist_ok=True)

def load_dataset(dataset_name, batch_size=32, split=None, with_info=False):
    """
    Loads a TensorFlow dataset with preprocessing, downloading it only if necessary.

    Args:
        dataset_name (str): Name of the dataset to load.
        batch_size (int): Batch size for training.
        split (list or str): Dataset splits to load.
        with_info (bool): Whether to return dataset info.

    Returns:
        Tuple of (train_ds, test_ds) or (train_ds, test_ds, info) if with_info=True
    """
    # Check if dataset is already downloaded
    dataset_path = os.path.join(DATASET_DIR, dataset_name)
    if os.path.exists(dataset_path):
        print(f"✅ Dataset '{dataset_name}' found locally. Using cached version.")
    else:
        print(f"⬇️  Downloading dataset '{dataset_name}' to {DATASET_DIR}...")

    dataset, info = tfds.load(dataset_name, split=split, as_supervised=True, with_info=True, data_dir=DATASET_DIR)

    def preprocess(image, label):
        """Normalize images to [0,1] range"""
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image, label

    if isinstance(dataset, dict):  # Multiple splits
        datasets = [dataset[s].map(preprocess).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE) for s in split]
    elif isinstance(dataset, list):  # If multiple splits are returned as a list
        datasets = [d.map(preprocess).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE) for d in dataset]
    else:  # Single split
        datasets = dataset.map(preprocess).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return (*datasets, info) if with_info else tuple(datasets)

def dataset_basic_statistics(dataset):
    """
    Computes basic statistics about the dataset, including class distribution.

    Args:
        dataset (tf.data.Dataset): Dataset to analyze.

    Returns:
        None
    """
    print("\n📊 Dataset Statistics:")

    class_counts = collections.defaultdict(int)
    total_samples = 0
    image_shapes = set()

    for images, labels in dataset:
        labels = labels.numpy()
        total_samples += len(labels)
        for label in labels:
            class_counts[label] += 1
        for img in images:
            image_shapes.add(img.shape)

    print(f"🔹 Total Samples: {total_samples}")
    print(f"🔹 Unique Image Shapes: {image_shapes}")
    print("🔹 Class Distribution:")
    for label, count in sorted(class_counts.items()):
        print(f"  - Class {label}: {count} samples")
