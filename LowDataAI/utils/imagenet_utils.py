# utils/imagenet_utils.py


from __future__ import annotations
import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def extract_numpy(ds: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Teljes (image, label) tf.data → (X, y) numpy.
    Figyelem: memóriát igényel, nagy halmazon csak óvatosan!
    - X alakja: [N, H, W, C] float32 (ha előtte már normalizálva volt)
    - y alakja: [N] int
    """
    X_parts, y_parts = [], []
    for images, labels in ds:
        X_parts.append(images.numpy())
        y_parts.append(labels.numpy())
    X = np.concatenate(X_parts, axis=0) if X_parts else np.empty((0,))
    y = np.concatenate(y_parts, axis=0) if y_parts else np.empty((0,), dtype=np.int32)
    return X, y


def per_class_equal_subset(
    train_ds: tf.data.Dataset,
    num_classes: int,
    fraction: float = 0.1,
    seed: int = 42,
    batch_size: int = 32,
) -> tf.data.Dataset:
    """
    Egyenletes mintavétel minden osztályból a 'fraction' arány szerint.
    Megközelítés:
      1) materializálás numpy-ba (gyors, de memóriás),
      2) per-osztály indexekből mintavétel,
      3) képek uint8 tömörítése → CPU-n from_tensor_slices → futás közben visszaskálázás.
    Így elkerülhető a nagy _EagerConst allokáció GPU-n.

    Visszatér: tf.data.Dataset (shuffle → batch → prefetch)
    """
    rng = np.random.default_rng(seed)

    # 1) Materializálás numpy-ba
    X, y = extract_numpy(train_ds.unbatch().batch(1024))

    if X.size == 0:
        raise ValueError("per_class_equal_subset: üres train_ds érkezett.")

    # 2) Indexválasztás osztályonként
    total = X.shape[0]
    target = max(1, int(total * float(fraction)))
    per_class_target = max(1, target // max(1, num_classes))

    idx_by_cls = {c: np.where(y == c)[0] for c in range(num_classes)}
    chosen_idx = []
    for c in range(num_classes):
        pool = idx_by_cls.get(c, np.array([], dtype=int))
        if pool.size == 0:
            continue
        n = min(per_class_target, pool.size)
        chosen_idx.extend(rng.choice(pool, n, replace=False))
    rng.shuffle(chosen_idx)

    # 3) Vágás + tömörítés (float32 [0,1] → uint8 [0..255])
    Xs = X[chosen_idx]
    ys = y[chosen_idx]
    Xs = (np.clip(Xs, 0.0, 1.0) * 255.0).astype(np.uint8)

    # 4) CPU-n dataset építés, futás közbeni visszaskálázás
    with tf.device("/CPU:0"):
        ds = tf.data.Dataset.from_tensor_slices((Xs, ys))

    def _to_float32(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    ds = (
        ds.map(_to_float32, num_parallel_calls=AUTOTUNE)
          .shuffle(1024, reshuffle_each_iteration=True)
          .batch(batch_size)
          .prefetch(AUTOTUNE)
    )
    return ds


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15
) -> float:
    if probs.ndim != 2:
        raise ValueError(f"compute_ece: probs dim != 2 (got {probs.shape})")
    if labels.ndim != 1 or labels.shape[0] != probs.shape[0]:
        raise ValueError("compute_ece: labels alakja [N] és N-nek egyeznie kell a probs első dimenziójával.")

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if not np.any(mask):
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.mean()) * abs(bin_conf - bin_acc)

    return float(ece)
