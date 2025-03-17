import tensorflow as tf
import numpy as np
import collections

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
