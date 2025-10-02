# utils/imagenet_utils.py
import glob
import os
import random
import shutil
import zipfile
from pathlib import Path

import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE

def _standard_preprocess(image, label, image_size=(64, 64)):
    image = tf.image.resize(image, image_size, method=tf.image.ResizeMethod.BICUBIC)
    image = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]
    return image, label

def _augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    return image, label

def _build_ds(ds, image_size, batch_size, shuffle=True, augment=False,
              shuffle_buffer=128, prefetch_size=tf.data.AUTOTUNE):
    # 1) standard preprocess
    ds = ds.map(lambda x, y: _standard_preprocess(x, y, image_size=image_size),
                num_parallel_calls=AUTOTUNE)
    # 2) BATCH ELŐTT NEM SHUFFLE-zunk (vagy nagyon kicsi bufferrel)
    ds = ds.batch(batch_size)

    # 3) shuffle BATCH UTÁN (kicsi bufferrel is elég)
    if shuffle:
        ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)

    if augment:
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)

    ds = ds.prefetch(prefetch_size)
    return ds


def load_from_directory(
    root_dir: str,
    image_size=(64, 64),
    batch_size=32,
    validation_split=None,
    seed=42,
    subset_for_train="training",
    follow_links=False
):
    train_dir = os.path.join(root_dir, "train")
    val_dir = os.path.join(root_dir, "val")
    # 1) Osztályok csak a train alapján:
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Missing train dir: {train_dir}")

    class_names = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])
    num_classes = len(class_names)

    def make_ds(directory, subset=None, use_split=False):
        if not os.path.isdir(directory):
            return None
        ds = tf.keras.utils.image_dataset_from_directory(
            directory,
            label_mode="int",
            image_size=image_size,
            batch_size=None,        # előbb map, aztán batch
            shuffle=True,
            seed=seed,
            validation_split=validation_split if use_split else None,
            subset=subset if use_split else None,
            follow_links=follow_links,
            class_names=class_names # <-- fix: train mapping mindenhol
        )
        ds = _build_ds(ds, image_size=image_size, batch_size=batch_size, shuffle=True, augment=False)
        return ds

    if validation_split:
        # Trainből vágjuk a val-t → mapping automatikusan egységes
        train_ds = make_ds(train_dir, subset=subset_for_train, use_split=True)
        val_ds   = make_ds(train_dir, subset="validation", use_split=True)
    else:
        train_ds = make_ds(train_dir)
        val_ds   = make_ds(val_dir)

    # Tiny-ImageNet "test" nem class-mappás → hagyjuk None-on
    test_ds = None

    if train_ds is None or val_ds is None:
        raise FileNotFoundError(
            f"Expected train/val folders under {root_dir} with class subdirs. "
            f"Got train_ds={train_ds is not None}, val_ds={val_ds is not None}."
        )

    return train_ds, val_ds, test_ds, class_names, num_classes


def reorganize_tiny_imagenet_val(val_dir: str, annotations_file: str):
    """
    Tiny-ImageNet 'val' mappa egybe van, ezt klasszokra bontjuk.
    - val_dir: .../val/images
    - annotations_file: .../val/val_annotations.txt
    Csak egyszer kell lefuttatni; készít class alkönyvtárakat és odamozgatja a képeket.
    """
    import shutil
    with open(annotations_file, "r") as f:
        lines = [l.strip().split() for l in f.readlines()]
    mapping = {img: cls for (img, cls, *_rest) in lines}

    images_dir = val_dir
    parent = os.path.dirname(images_dir)  # .../val
    for img, cls in mapping.items():
        cls_dir = os.path.join(parent, cls)
        os.makedirs(cls_dir, exist_ok=True)
        src = os.path.join(images_dir, img)
        dst = os.path.join(cls_dir, img)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.move(src, dst)


def extract_numpy(ds):
    """Teljes ds → (X, y) numpy (vigyázat: memóriás, de label-budget kicsi)."""
    X, y = [], []
    for batch in ds:
        images, labels = batch
        X.append(images.numpy())
        y.append(labels.numpy())
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y


def per_class_equal_subset(train_ds, num_classes, fraction=0.1, seed=42, batch_size=32):
    """
    Nem streamelünk: materializálunk, de:
      - előbb kiválasztjuk az indexeket,
      - a képeket uint8-ban tároljuk (4× kisebb),
      - a from_tensor_slices kifejezetten CPU-ra kerül.
    """
    rng = np.random.default_rng(seed)

    # 1) Materializálás numpy-ba (most még float32 [0,1])
    X, y = extract_numpy(train_ds.unbatch().batch(1024))

    # 2) Indexválasztás osztályonként
    total = X.shape[0]
    target = int(total * fraction)
    per_class_target = max(1, target // num_classes)
    idx_by_cls = {c: np.where(y == c)[0] for c in range(num_classes)}
    chosen_idx = []
    for c in range(num_classes):
        pool = idx_by_cls[c]
        n = min(per_class_target, len(pool))
        if n > 0:
            chosen_idx.extend(rng.choice(pool, n, replace=False))
    rng.shuffle(chosen_idx)

    # 3) Vágás és TÖMÖRÍTÉS: float32 → uint8 (0..255)
    Xs = X[chosen_idx]
    ys = y[chosen_idx]
    # ha X float32 [0,1], akkor:
    Xs = (np.clip(Xs, 0.0, 1.0) * 255.0).astype(np.uint8)

    # 4) Dataset: CPU-n hozzuk létre, hogy ne menjen fel 2.3 GiB _EagerConst a GPU-ra
    with tf.device("/CPU:0"):
        ds = tf.data.Dataset.from_tensor_slices((Xs, ys))

    # 5) Visszaskálázás csak itt, apró batch-ekben → GPU-ra már kicsi szeletek mennek
    def _to_float32(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    ds = ds.map(_to_float32, num_parallel_calls=AUTOTUNE)
    ds = ds.shuffle(1024, reshuffle_each_iteration=True).batch(batch_size).prefetch(AUTOTUNE)
    return ds

# #TODO ez nem jó nagyokra
# def per_class_equal_subset(train_ds, num_classes, fraction=0.1, seed=42):
#     """
#     Egyenlő számú mintát vesz minden osztályból a kívánt frakciónak megfelelően.
#     - train_ds: batchelt tf.data
#     - fraction: pl. 0.1, 0.2, 0.3, 0.5
#     """
#     rng = np.random.default_rng(seed)
#     X, y = extract_numpy(train_ds.unbatch().batch(1024))  # gyorsabb aggregálás
#     # elemszám becslés
#     total = X.shape[0]
#     target = int(total * fraction)
#     per_class_target = max(1, target // num_classes)
#
#     idx_by_cls = {c: np.where(y == c)[0] for c in range(num_classes)}
#     chosen_idx = []
#     for c in range(num_classes):
#         pool = idx_by_cls[c]
#         n = min(per_class_target, len(pool))
#         chosen_idx.extend(rng.choice(pool, n, replace=False))
#     rng.shuffle(chosen_idx)
#     Xs = X[chosen_idx]
#     ys = y[chosen_idx]
#
#     ds = tf.data.Dataset.from_tensor_slices((Xs, ys)).shuffle(4096, seed=seed).batch(32).prefetch(AUTOTUNE)
#     return ds

# def per_class_equal_subset(train_ds, num_classes, fraction=0.1, seed=42, batch_size=32):
#     # Backward-compat: csak továbbhívjuk a streames verziót
#     return balanced_subset_from_stream(
#         train_ds, num_classes, fraction=fraction, batch_size=batch_size, seed=seed
#     )


def label_budget_presets(train_ds, num_classes, presets=(0.1, 0.2, 0.3, 0.5), seed=42):
    """Visszaad egy dictet frakció → subset_ds."""
    return {p: per_class_equal_subset(train_ds, num_classes, p, seed=seed) for p in presets}


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    """
    Expected Calibration Error (softmax probs + int labels).
    probs: [N, C], labels: [N]
    """
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


TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

def ensure_tiny_imagenet(root_dir: str):
    """
    Letölti és kicsomagolja a Tiny-ImageNet-200-at a root_dir-be, ha még nincs ott.
    A végén gondoskodik róla, hogy a 'val' mappa class-onként legyen szétszedve.
    Elrendezés: <root_dir>/tiny-imagenet-200/{train,val,test}
    """
    root = Path(root_dir)
    target = root / "tiny-imagenet-200"
    train_dir = target / "train"
    val_dir = target / "val"
    images_dir = val_dir / "images"
    annotations = val_dir / "val_annotations.txt"

    if train_dir.is_dir() and val_dir.is_dir() and not images_dir.is_dir():
        # már korábban reorganize-olva
        return

    if not target.is_dir():
        root.mkdir(parents=True, exist_ok=True)
        print(f"⬇️  Downloading Tiny-ImageNet to {root} ...")
        zip_path = tf.keras.utils.get_file(
            fname="tiny-imagenet-200.zip",
            origin=TINY_IMAGENET_URL,
            cache_dir=str(root),
            cache_subdir=".",
            extract=False,
        )
        print("📦 Extracting...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(root))

    # Ha még egyben van a val/images, szervezzük szét
    if images_dir.is_dir() and annotations.is_file():
        print("🗂️  Reorganizing Tiny-ImageNet val/ into class folders...")
        reorganize_tiny_imagenet_val(str(images_dir), str(annotations))
        # az images mappa kiürül, ez oké

    images_dir_path = Path(images_dir)
    if images_dir_path.exists():
        print("🧹 Removing empty val/images folder to avoid extra class.")
        shutil.rmtree(images_dir_path, ignore_errors=True)

def ensure_imagenet100_root(root_dir: str):
    """
    Csak ellenőriz: felhasználó által előkészített ImageNet-100 folder layout kell.
    Elvárt: <root_dir>/imagenet-100/{train,val,(test)}/<class>/*.jpg
    """
    target = Path(root_dir) / "imagenet-100"
    if not (target / "train").is_dir() or not (target / "val").is_dir():
        raise FileNotFoundError(
            f"ImageNet-100 not found under {target}. "
            "Create folders train/val/(test) with class subdirs."
        )

def balanced_subset_from_stream(ds, num_classes, fraction, batch_size, seed=1337):
    # Elemenkénti stream a filterhez (nálad _build_ds már batchel)
    ds = ds.unbatch()

    # Tiny-ImageNet ~500 kép/osztály → ebből vesszük a frakciót
    per_class_k = max(1, int(round(500 * float(fraction))))

    # Stabil lezárás: külső függvénnyel gyártjuk a predikátumot
    def make_class_filter(cls_id: int):
        cls_t = tf.constant(cls_id, dtype=tf.int32)
        return lambda x, y: tf.equal(y, cls_t)  # ha one-hot lenne: tf.equal(tf.argmax(y, -1), cls_t)

    # Osztályonként kiválasztunk k mintát, majd összefűzzük
    subs = []
    for cls in range(num_classes):
        sub_c = ds.filter(make_class_filter(cls)).take(per_class_k)
        subs.append(sub_c)

    subset = subs[0]
    for s in subs[1:]:
        subset = subset.concatenate(s)

    # Shuffle → batch → prefetch
    subset = subset.shuffle(4096, seed=seed, reshuffle_each_iteration=False)
    subset = subset.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return subset

_VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

def per_class_file_subset(train_dir, class_names, fraction=0.1, seed=42):
    """
    train/<class>/images/ alól választ képeket. Windows-on a glob case-insensitive,
    ezért előbb mindent összeszedünk, majd .lower()-rel EXT alapján szűrünk és DEDUP-olunk.
    """
    rng = random.Random(seed)
    filepaths, labels = [], []

    for cls_idx, cls_name in enumerate(class_names):
        cls_img_dir = os.path.join(train_dir, cls_name, "images")
        if not os.path.isdir(cls_img_dir):
            continue

        # 1) összes fájl egyszer
        all_paths = glob.glob(os.path.join(cls_img_dir, "*"))

        # 2) szűrés érvényes képekre + dedup
        uniq = {}
        for p in all_paths:
            if not os.path.isfile(p):
                continue
            ext = os.path.splitext(p)[1].lower()
            if ext in _VALID_EXTS:
                # normcase + lower kulccsal dedup (Windowsnál fontos)
                key = os.path.normcase(p).lower()
                uniq[key] = p
        img_files = list(uniq.values())

        if not img_files:
            continue

        # 3) per-class mintaszám
        n_select = max(1, int(len(img_files) * float(fraction)))
        chosen = rng.sample(img_files, n_select)

        filepaths.extend(chosen)
        labels.extend([cls_idx] * len(chosen))

    if not filepaths:
        raise RuntimeError("No images found under train/<class>/images with valid extensions.")

    return filepaths, labels

def build_dataset_from_files(filepaths, labels, image_size, batch_size, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))

    def _load_img(path, label):
        img_bytes = tf.io.read_file(path)
        # Formátum-agnosztikus (JPEG/PNG/GIF/BMP), animált gifet egy képkockára lapít
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])  # statikus alak a későbbi resize-hoz
        return img, label

    ds = ds.map(_load_img, num_parallel_calls=AUTOTUNE)
    ds = _build_ds(ds, image_size=image_size, batch_size=batch_size, shuffle=shuffle, augment=False)
    return ds
