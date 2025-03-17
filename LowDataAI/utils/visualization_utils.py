import matplotlib.pyplot as plt
import numpy as np

def plot_random_images(dataset, num_images=9):
    """
    Plots a grid of sample images from the dataset.

    Args:
        dataset (tf.data.Dataset): The dataset to sample images from.
        num_images (int): Number of images to display.

    Returns:
        None
    """
    plt.figure(figsize=(8, 8))
    images, labels = next(iter(dataset.take(1)))

    for i in range(min(num_images, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(f"Class: {labels[i].numpy()}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
