import collections
import math
import os
import time
import json
import tensorflow as tf
from matplotlib import pyplot as plt


# import tensorflow_datasets as tfds
#
#
# # Set a custom directory for storing datasets
# DATASET_DIR = os.path.join("..", "data", "tfds_data")
# os.makedirs(DATASET_DIR, exist_ok=True)
#
# def load_dataset(dataset_name, batch_size=32, split=None, with_info=False):
#     # Check if dataset is already downloaded
#     dataset_path = os.path.join(DATASET_DIR, dataset_name)
#     if os.path.exists(dataset_path):
#         print(f"✅ Dataset '{dataset_name}' found locally. Using cached version.")
#     else:
#         print(f"⬇️  Downloading dataset '{dataset_name}' to {DATASET_DIR}...")
#
#     dataset, info = tfds.load(dataset_name, split=split, as_supervised=True, with_info=True, data_dir=DATASET_DIR)
#
#     def preprocess(image, label):
#         """Normalize images to [0,1] range"""
#         image = tf.image.convert_image_dtype(image, tf.float32)
#         return image, label
#
#     if isinstance(dataset, dict):  # Multiple splits
#         datasets = [dataset[s].map(preprocess).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE) for s in split]
#     elif isinstance(dataset, list):  # If multiple splits are returned as a list
#         datasets = [d.map(preprocess).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE) for d in dataset]
#     else:  # Single split
#         datasets = dataset.map(preprocess).shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
#
#     return (*datasets, info) if with_info else tuple(datasets)

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

def extract_numpy_data(dataset):
    """Convert a batched (image, label) dataset into full numpy arrays."""
    images, labels = [], []
    for img_batch, label_batch in dataset:
        for i in range(len(img_batch)):
            images.append(img_batch[i].numpy())
            labels.append(label_batch[i].numpy())
    return np.array(images), np.array(labels)

def save_dataset_snapshot(dataset, name_prefix="quartered_animals", output_dir="../results/dataset_snapshots"):
    os.makedirs(output_dir, exist_ok=True)
    images, labels = extract_numpy_data(dataset)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{name_prefix}_{timestamp}.npz"
    filepath = os.path.join(output_dir, filename)
    np.savez_compressed(filepath, images=images, labels=labels)
    print(f"📁 Dataset snapshot saved to {filepath}")

def preview_random_samples(
    train_ds: tf.data.Dataset,
    labels_json_path: str,          # <-- JSON útvonal érkezik
    out_path: str,
    n: int = 5,
    seed: int = 42
) -> None:
    rng = np.random.default_rng(seed)

    # --- Labels.json beolvasás és index->WNID->név feloldás ---
    with open(labels_json_path, "r", encoding="utf-8") as f:
        wnid_to_name = json.load(f)          # {wnid: human name}
    class_wnids_sorted = sorted(wnid_to_name.keys())

    def label_to_text(lbl: int) -> str:
        if 0 <= lbl < len(class_wnids_sorted):
            wnid = class_wnids_sorted[lbl]
            human = wnid_to_name.get(wnid, wnid)
            return f"{wnid}"
        return str(lbl)

    # --- mintavétel + normalizálás/diagnosztika (változatlan logika) ---
    samples = []
    for img, lab in train_ds.unbatch().take(300):
        samples.append((img, lab))
    if not samples:
        print("[preview] Üres dataset – nincs mit rajzolni.")
        return

    n = max(3, min(int(n), 99))
    idx = rng.choice(np.arange(len(samples)), size=n, replace=False)

    proc_imgs, titles = [], []
    print("[preview] ---- minták diagnosztika ----")
    for i in idx:
        img_t, lab_t = samples[i]
        img = img_t.numpy()
        lab = int(lab_t.numpy())

        if img.ndim == 4 and img.shape[0] == 1: img = img[0]
        if img.ndim == 2: img = np.stack([img, img, img], axis=-1)
        if img.ndim != 3: raise ValueError(f"Várt (H,W,C), kaptam {img.shape}")
        if img.shape[-1] == 1: img = np.repeat(img, 3, axis=-1)
        if img.shape[-1] != 3 and img.shape[0] == 3: img = np.transpose(img, (1,2,0))

        arr = img.astype(np.float32)
        if np.isnan(arr).any() or np.isinf(arr).any():
            print(f"[preview] WARN: NaN/Inf (idx={i}) – kihagyom.")
            continue
        if arr.max() > 1.5:
            arr = np.clip(arr, 0.0, 255.0) / 255.0

        print(f"[preview] i={i} shape={arr.shape} range=[{arr.min():.4f},{arr.max():.4f}] label={lab}")
        proc_imgs.append(arr)
        titles.append(label_to_text(lab))

    if not proc_imgs:
        print("[preview] Nem maradt minta.")
        return

    cols = min(5, len(proc_imgs)); rows = math.ceil(len(proc_imgs)/cols)
    plt.figure(figsize=(cols*2.6, rows*2.6))
    k = 1
    for img, title in zip(proc_imgs, titles):
        plt.subplot(rows, cols, k)
        plt.imshow(img, interpolation="nearest", aspect="equal")
        plt.title(title, fontsize=8)
        plt.axis("off"); k += 1
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout(); plt.savefig(out_path, dpi=140); plt.close()
    print(f"[preview] Mentve: {out_path}")

