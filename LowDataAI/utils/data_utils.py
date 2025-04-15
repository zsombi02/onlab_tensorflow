import collections
import os
import tensorflow as tf
import tensorflow_datasets as tfds

# Set a custom directory for storing datasets
DATASET_DIR = os.path.join("..", "data", "tfds_data")
os.makedirs(DATASET_DIR, exist_ok=True)

def load_dataset(dataset_name, batch_size=32, split=None, with_info=False):
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


import tensorflow as tf
import numpy as np

def create_subset(dataset, num_classes=10, subset_fraction=0.5, seed=42):

    all_images = []
    all_labels = []

    for batch in dataset:
        images, labels = batch
        all_images.append(images)
        all_labels.append(labels)

    images = tf.concat(all_images, axis=0).numpy()
    labels = tf.concat(all_labels, axis=0).numpy()

    # Collect indices by class
    indices_by_class = {i: np.where(labels == i)[0] for i in range(num_classes)}

    # Calculate number of samples per class for the subset
    np.random.seed(seed)
    samples_per_class = int(len(labels) * subset_fraction / num_classes)

    subset_indices = []
    for cls, indices in indices_by_class.items():
        chosen = np.random.choice(indices, samples_per_class, replace=False)
        subset_indices.extend(chosen)

    # Shuffle the collected indices
    subset_indices = np.array(subset_indices)
    np.random.shuffle(subset_indices)

    # Slice the image and label arrays
    subset_images = images[subset_indices]
    subset_labels = labels[subset_indices]

    # Rebuild the dataset
    subset_ds = tf.data.Dataset.from_tensor_slices((subset_images, subset_labels))
    subset_ds = subset_ds.batch(32).prefetch(tf.data.AUTOTUNE)

    return subset_ds


def create_subset_fewer_frogs(dataset, num_classes=10, subset_fraction=0.25, frog_class=6, frog_fraction=0.5, seed=42):
    all_images = []
    all_labels = []

    for batch in dataset:
        images, labels = batch
        all_images.append(images)
        all_labels.append(labels)

    images = tf.concat(all_images, axis=0).numpy()
    labels = tf.concat(all_labels, axis=0).numpy()

    indices_by_class = {i: np.where(labels == i)[0] for i in range(num_classes)}
    total_target = int(len(labels) * subset_fraction)
    base_per_class = total_target // num_classes

    samples_per_class = {i: base_per_class for i in range(num_classes)}
    samples_per_class[frog_class] = int(base_per_class * frog_fraction)

    subset_indices = []
    np.random.seed(seed)
    for cls, indices in indices_by_class.items():
        n = samples_per_class[cls]
        chosen = np.random.choice(indices, n, replace=False)
        subset_indices.extend(chosen)

    np.random.shuffle(subset_indices)
    subset_images = images[subset_indices]
    subset_labels = labels[subset_indices]

    subset_ds = tf.data.Dataset.from_tensor_slices((subset_images, subset_labels))
    subset_ds = subset_ds.batch(32).prefetch(tf.data.AUTOTUNE)

    return subset_ds


def create_subset_more_critical_classes(dataset, num_classes=10, subset_fraction=0.25, boost_classes=[2, 3, 5, 7], boost_factor=2, seed=42):
    all_images = []
    all_labels = []

    for batch in dataset:
        images, labels = batch
        all_images.append(images)
        all_labels.append(labels)

    images = tf.concat(all_images, axis=0).numpy()
    labels = tf.concat(all_labels, axis=0).numpy()

    indices_by_class = {i: np.where(labels == i)[0] for i in range(num_classes)}
    total_target = int(len(labels) * subset_fraction)
    base_per_class = total_target // (num_classes + len(boost_classes) * (boost_factor - 1))

    samples_per_class = {i: base_per_class for i in range(num_classes)}
    for cls in boost_classes:
        samples_per_class[cls] = base_per_class * boost_factor

    subset_indices = []
    np.random.seed(seed)
    for cls, indices in indices_by_class.items():
        n = samples_per_class[cls]
        chosen = np.random.choice(indices, n, replace=False)
        subset_indices.extend(chosen)

    np.random.shuffle(subset_indices)
    subset_images = images[subset_indices]
    subset_labels = labels[subset_indices]

    subset_ds = tf.data.Dataset.from_tensor_slices((subset_images, subset_labels))
    subset_ds = subset_ds.batch(32).prefetch(tf.data.AUTOTUNE)

    return subset_ds



def augment_image(image, label):
    """Simple augmentations: random flip, rotation, brightness"""
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.rot90(image, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    return image, label


def create_augmented_dataset(dataset, factor=2, seed=42):

    tf.random.set_seed(seed)
    augmented_versions = []

    # Unbatch to work on single (image, label) pairs
    dataset = dataset.unbatch()

    # First: original dataset
    augmented_versions.append(dataset)

    # Additional versions with augmentation
    for _ in range(factor - 1):
        augmented = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
        augmented_versions.append(augmented)

    # Combine and rebatch
    full_dataset = augmented_versions[0]
    for ds in augmented_versions[1:]:
        full_dataset = full_dataset.concatenate(ds)

    # Rebatch and prefetch
    full_dataset = full_dataset.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
    return full_dataset

