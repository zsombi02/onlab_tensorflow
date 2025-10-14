# data/imagenet.py
import os

from utils.imagenet_data_utils import (
    ensure_imagenet100_root, ensure_imagenet100_from_kaggle,
    build_class_index_with_labels_json, build_dataset_from_files, per_class_file_subset,
)

DATA_ROOT = "../data/imagenet_subsets"
KAGGLE_TARGET_PARENT = "../data/imagenet_subsets"

def quick_label_check(paths, labels, wnid_to_idx, n=50):
    import random, os
    sample = random.sample(list(zip(paths, labels)), k=min(n, len(paths)))
    for p, y in sample:
        wnid = os.path.basename(os.path.dirname(p))
        assert wnid_to_idx[wnid] == y, f"Label mismatch: {p} -> {wnid_to_idx[wnid]} != {y}"


def load_imagenet100_kaggle_budget(pct=0.1, batch_size=64, image_size=(224,224), seed=42):
    root = ensure_imagenet100_from_kaggle(KAGGLE_TARGET_PARENT)
    ensure_imagenet100_root(KAGGLE_TARGET_PARENT)
    labels_json_path = os.path.join(root, "Labels.json")
    class_names, wnid_to_idx, _, _ = build_class_index_with_labels_json(root, labels_json_path)

    if not os.path.isfile(labels_json_path):
        labels_json_path = os.path.join(os.path.dirname(root), "Labels.json")
    class_names, *_ = build_class_index_with_labels_json(root, labels_json_path)
    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")
    filepaths, labels = per_class_file_subset(train_dir, class_names, fraction=float(pct), seed=seed)
    val_filepaths, val_labels = per_class_file_subset(val_dir, class_names, fraction=1.0, seed=seed)
    quick_label_check(filepaths, labels, wnid_to_idx, n=200)
    quick_label_check(val_filepaths,   val_labels,   wnid_to_idx, n=200)
    train_sub = build_dataset_from_files(filepaths, labels, image_size=image_size, batch_size=batch_size, shuffle=True, augment=True)
    val_sub = build_dataset_from_files(val_filepaths, val_labels, image_size=image_size, batch_size=batch_size, shuffle=False, augment=False)


    return train_sub, val_sub


def load_imagenet50_from_imagenet100(pct=0.1, batch_size=64, image_size=(224, 224), seed=42):
    root = ensure_imagenet100_from_kaggle(KAGGLE_TARGET_PARENT)
    ensure_imagenet100_root(KAGGLE_TARGET_PARENT)

    labels_json_path = os.path.join(root, "Labels.json")
    if not os.path.isfile(labels_json_path):
        labels_json_path = os.path.join(os.path.dirname(root), "Labels.json")

    full_class_names, wnid_to_idx, _, _ = build_class_index_with_labels_json(root, labels_json_path)

    TARGET_NUM_CLASSES = 50
    if len(full_class_names) < TARGET_NUM_CLASSES:
        raise ValueError(
            f"Csak {len(full_class_names)} osztályt találtam. Nem tudok {TARGET_NUM_CLASSES}-t kiválasztani.")

    class_names_50 = full_class_names[:TARGET_NUM_CLASSES]
    print(f"✅ ImageNet-100 alapú ImageNet-50 betöltve. WNID tartomány: {class_names_50[0]} - {class_names_50[-1]}")

    train_dir = os.path.join(root, "train")
    val_dir = os.path.join(root, "val")

    filepaths, labels = per_class_file_subset(train_dir, class_names_50, fraction=float(pct), seed=seed)

    val_filepaths, val_labels = per_class_file_subset(val_dir, class_names_50, fraction=1.0, seed=seed)

    train_sub = build_dataset_from_files(filepaths, labels, image_size=image_size, batch_size=batch_size, shuffle=True,
                                         augment=True)
    val_sub = build_dataset_from_files(val_filepaths, val_labels, image_size=image_size, batch_size=batch_size,
                                       shuffle=False, augment=False)

    return train_sub, val_sub