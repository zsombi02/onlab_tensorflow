# data/imagenet_data_utils.py
import os, glob, json, random, shutil, zipfile
from pathlib import Path
import tensorflow as tf
import numpy as np
import kagglehub  # csak az ImageNet-100 Kaggle mirrorhoz

AUTOTUNE = tf.data.AUTOTUNE
_VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

# ---------- Közös tf.data építők ----------

def _standard_preprocess(image, label, image_size):
    image = tf.image.resize(image, image_size, method=tf.image.ResizeMethod.BICUBIC)
    image = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]
    return image, label

def build_ds_from_images(ds, image_size, batch_size, shuffle=True,
                         shuffle_buffer=16384, augment=False):
    # 1. Előfeldolgozás minden képen (map)
    ds = ds.map(lambda x, y: _standard_preprocess(x, y, image_size), num_parallel_calls=AUTOTUNE)

    # 2. Keverés (ha kell)
    if shuffle:
        ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)

    # 3. Batch-ekbe rendezés. FONTOS: Ez most az augmentáció ELŐTT van!
    ds = ds.batch(batch_size)

    # 4. Augmentáció az egész batch-en (GPU-n)
    if augment:
        # Készítünk egy mini-modellt csak az augmentációhoz
        augmentation_layers = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.40),
            tf.keras.layers.RandomTranslation(height_factor=0.15, width_factor=0.15)
        ], name="augmentation_pipeline")

        # Alkalmazzuk ezt a mini-modellt a batchelt képekre
        ds = ds.map(lambda x, y: (augmentation_layers(x, training=True), y),
                    num_parallel_calls=AUTOTUNE)

    # 5. Adatok előtöltése a következő lépéshez
    return ds.prefetch(AUTOTUNE)

def build_dataset_from_files(filepaths, labels, image_size, batch_size, shuffle=True, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    def _load(path, label):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        return img, label
    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    return build_ds_from_images(ds, image_size, batch_size, shuffle=shuffle, augment=augment)


def per_class_file_subset(train_dir, class_names, fraction=0.1, seed=42):
    rng = random.Random(seed)
    filepaths, labels = [], []

    # Konvertálás float-tá az összehasonlításhoz
    fraction = float(fraction)

    for cls_idx, cls_name in enumerate(class_names):
        cls_dir = os.path.join(train_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue

        # Képfájlok gyűjtése (a dedikált logikát megtartva)
        all_paths = glob.glob(os.path.join(cls_dir, "*"))
        uniq = {}
        for p in all_paths:
            if not os.path.isfile(p):
                continue
            ext = os.path.splitext(p)[1].lower()
            if ext in _VALID_EXTS:
                uniq[os.path.normcase(p).lower()] = p
        img_files = list(uniq.values())

        if not img_files:
            continue

        if fraction >= 1.0:
            chosen = img_files
        else:
            # Eredeti Logika: Mintavétel (ha 0 < fraction < 1.0)
            n_select = max(1, int(len(img_files) * fraction))
            chosen = rng.sample(img_files, n_select)

        filepaths.extend(chosen)
        labels.extend([cls_idx] * len(chosen))

    if not filepaths:
        raise RuntimeError("No images found under train/<class> with valid extensions.")
    return filepaths, labels

# ---------- Labels.json + class_names ----------

def read_wnid_to_name(labels_json_path: str) -> dict[str, str]:
    with open(labels_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Labels.json must be a dict of WNID -> name.")
    return {str(k).strip(): str(v).strip() for k, v in data.items()}

def list_class_dirs(split_dir: str) -> set[str]:
    return {d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))}

def build_class_index_with_labels_json(root_dir: str, labels_json_path: str):
    train_dir = os.path.join(root_dir, "train")
    val_dir   = os.path.join(root_dir, "val")
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Missing split dir(s): {train_dir} / {val_dir}")
    wnid_to_name = read_wnid_to_name(labels_json_path)
    common = set(wnid_to_name.keys()) & list_class_dirs(train_dir) & list_class_dirs(val_dir)
    if not common:
        raise RuntimeError("No common WNIDs between Labels.json and train/val.")
    class_names = sorted(common)
    wnid_to_idx = {w:i for i,w in enumerate(class_names)}
    idx_to_wnid = {i:w for w,i in wnid_to_idx.items()}
    return class_names, wnid_to_idx, idx_to_wnid, wnid_to_name

# ---------- Tiny-ImageNet biztosítás ----------

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"

def reorganize_tiny_imagenet_val(val_dir: str, annotations_file: str):
    with open(annotations_file, "r") as f:
        mapping = {l.split()[0]: l.split()[1] for l in f.read().strip().splitlines()}
    images_dir = val_dir
    parent = os.path.dirname(images_dir)
    for img, cls in mapping.items():
        cls_dir = os.path.join(parent, cls)
        os.makedirs(cls_dir, exist_ok=True)
        src = os.path.join(images_dir, img)
        dst = os.path.join(cls_dir, img)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.move(src, dst)

def ensure_tiny_imagenet(root_dir: str):
    root = Path(root_dir)
    target = root / "tiny-imagenet-200"
    train_dir, val_dir = target/"train", target/"val"
    images_dir, annotations = val_dir/"images", val_dir/"val_annotations.txt"
    if train_dir.is_dir() and val_dir.is_dir() and not images_dir.is_dir():
        return
    if not target.is_dir():
        root.mkdir(parents=True, exist_ok=True)
        zip_path = tf.keras.utils.get_file(
            fname="tiny-imagenet-200.zip",
            origin=TINY_IMAGENET_URL,
            cache_dir=str(root),
            cache_subdir=".",
            extract=False,
        )
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(str(root))
    if images_dir.is_dir() and annotations.is_file():
        reorganize_tiny_imagenet_val(str(images_dir), str(annotations))
        shutil.rmtree(images_dir, ignore_errors=True)

# ---------- ImageNet-100 Kaggle mirror ----------

def _safe_link_or_copy(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)

def _list_image_files(p: Path):
    files = []
    for e in ("*.jpg","*.jpeg","*.png","*.bmp"):
        files.extend(glob.glob(str(p / e)))
    return [Path(f) for f in files]

def _has_class_subdirs(p: Path) -> bool:
    return p.is_dir() and any(d.is_dir() for d in p.iterdir())

def _discover_kaggle_shards(root: Path):
    train_shards, val_shards, labels_json = [], [], None
    cand = list(root.glob("Labels.json")) + list(root.rglob("Labels.json"))
    if cand: labels_json = cand[0]
    def _looks_like_class_shard(p: Path): return p.is_dir() and any(d.is_dir() for d in p.iterdir())
    for d in root.iterdir():
        if not d.is_dir(): continue
        n = d.name
        if n.startswith("train") and _looks_like_class_shard(d): train_shards.append(d)
        elif n.startswith("val"): val_shards.append(d)
    if (not train_shards) or (not val_shards):
        for d in root.iterdir():
            if not d.is_dir(): continue
            for s in d.iterdir():
                if not s.is_dir(): continue
                n = s.name
                if n.startswith("train") and _looks_like_class_shard(s): train_shards.append(s)
                elif n.startswith("val"): val_shards.append(s)
    if not train_shards: raise FileNotFoundError("train.* shard not found in Kaggle dataset.")
    if not val_shards:   raise FileNotFoundError("val.* shard not found in Kaggle dataset.")
    return sorted(set(train_shards)), sorted(set(val_shards)), labels_json

def _gather_union_wnids(shards: list[Path]) -> list[str]:
    wnids = set()
    for s in shards:
        for d in s.iterdir():
            if d.is_dir(): wnids.add(d.name)
    return sorted(wnids)

def _mirror_split_from_class_dirs(shards: list[Path], dst_split: Path, class_dirs_ref: list[str]):
    dst_split.mkdir(parents=True, exist_ok=True)
    total, mirrored = 0, set()
    for shard in shards:
        for cls_dir in sorted([d for d in shard.iterdir() if d.is_dir()]):
            wnid = cls_dir.name
            if class_dirs_ref is not None and wnid not in class_dirs_ref: continue
            tcls = dst_split / wnid
            tcls.mkdir(parents=True, exist_ok=True)
            for f in _list_image_files(cls_dir):
                _safe_link_or_copy(f, tcls / f.name)
                total += 1
            mirrored.add(wnid)
    return total, mirrored

def _mirror_val_with_labels(val_dir: Path, labels_json: Path, dst_split: Path, allowed_wnids: set[str]):
    with open(labels_json, "r", encoding="utf-8") as f:
        labels = json.load(f)
    if not isinstance(labels, dict):
        raise ValueError("Labels.json must be dict (WNID->name or file->wnid).")
    # ha WNID->name, nem tudunk file->wnid térképet, ez a mirror nem kell (nálad class-mappás a val)
    dst_split.mkdir(parents=True, exist_ok=True)
    total = 0
    for f in _list_image_files(val_dir):
        # itt nem használjuk labels-t, csak másolnánk – de nálad class-mappás, szóval ezt nem is hívjuk
        pass
    return total

def ensure_imagenet100_root(root_dir: str):
    target = Path(root_dir) / "imagenet-100"
    if not (target / "train").is_dir() or not (target / "val").is_dir():
        raise FileNotFoundError(f"ImageNet-100 not found under {target} (train/val with class subdirs required).")

def ensure_imagenet100_from_kaggle(target_parent: str) -> str:
    kaggle_root = Path(kagglehub.dataset_download("ambityga/imagenet100"))
    print(f"📥 Kaggle letöltés kész: {kaggle_root}")
    train_shards, val_shards, labels_json = _discover_kaggle_shards(kaggle_root)
    print("🔎 Train shardok:", [s.name for s in train_shards])
    print("🔎 Val   shardok:", [s.name for s in val_shards])
    if labels_json: print(f"🔎 Labels.json: {labels_json}")
    target_root = Path(target_parent) / "imagenet-100"
    train_dst, val_dst = target_root / "train", target_root / "val"
    train_dst.mkdir(parents=True, exist_ok=True)
    val_dst.mkdir(parents=True, exist_ok=True)
    all_wnids = _gather_union_wnids(train_shards)
    if len(all_wnids) > 100:
        print(f"⚠️ {len(all_wnids)} osztályt találtam; 100-ra vágom.")
        all_wnids = sorted(all_wnids)[:100]
    ref_set = set(all_wnids)
    n_train, mirrored_train = _mirror_split_from_class_dirs(train_shards, train_dst, class_dirs_ref=all_wnids)
    if ref_set - mirrored_train:
        miss = sorted(ref_set - mirrored_train)
        print(f"⚠️ Hiányzó train osztályok közül pár: {miss[:5]} (+{max(0,len(miss)-5)})")
    if all(_has_class_subdirs(s) for s in val_shards):
        n_val, _ = _mirror_split_from_class_dirs(val_shards, val_dst, class_dirs_ref=all_wnids)
    else:
        if not labels_json:
            raise RuntimeError("val shard nem class-mappás és Labels.json sincs; nem tudok kiosztani.")
        n_val = _mirror_val_with_labels(val_shards[0], labels_json, val_dst, allowed_wnids=ref_set)
    print(f"✅ ImageNet-100 build: {target_root}")
    print(f"   - train képek: {n_train:,}")
    print(f"   - val   képek: {n_val:,}")
    return str(target_root)

