# data/imagenet.py
from utils.imagenet_utils import load_from_directory, per_class_equal_subset, ensure_tiny_imagenet, \
    ensure_imagenet100_root, per_class_file_subset, build_dataset_from_files, ensure_imagenet100_from_kaggle
import os

DATA_ROOT = "../data/imagenet_subsets"

def load_tiny_imagenet(batch_size=64, image_size=(64,64)):
    root = os.path.join(DATA_ROOT, "tiny-imagenet-200")
    # 👇 auto letöltés + val/ reorganize
    ensure_tiny_imagenet(DATA_ROOT)
    train_ds, val_ds, test_ds, class_names, num_classes = load_from_directory(
        root, image_size=image_size, batch_size=batch_size, validation_split=None
    )
    return train_ds, val_ds, num_classes

# def load_tiny_imagenet_budget(pct=0.1, batch_size=64, image_size=(64,64), seed=42):
#     train_full, val_ds, num_classes = load_tiny_imagenet(batch_size=batch_size, image_size=image_size)
#     train_sub = per_class_equal_subset(train_full, num_classes, fraction=pct, seed=seed, batch_size=batch_size)
#     return train_sub, val_ds

def load_tiny_imagenet_budget(pct=0.1, batch_size=64, image_size=(64,64), seed=42):
    root = os.path.join(DATA_ROOT, "tiny-imagenet-200")
    ensure_tiny_imagenet(DATA_ROOT)
    train_dir = os.path.join(root, "train")

    # class_names fix: train alapján
    class_names = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])
    num_classes = len(class_names)

    # fájllista + címkék mintavételezése
    filepaths, labels = per_class_file_subset(train_dir, class_names, fraction=pct, seed=seed)

    # subset dataset
    train_sub = build_dataset_from_files(filepaths, labels,
                                         image_size=image_size,
                                         batch_size=batch_size,
                                         shuffle=True)

    # validáció ugyanúgy
    _, val_ds, _, _, _ = load_from_directory(root,
                                             image_size=image_size,
                                             batch_size=batch_size,
                                             validation_split=None)

    return train_sub, val_ds

def load_imagenet100(batch_size=64, image_size=(224,224)):
    root = os.path.join(DATA_ROOT, "imagenet-100")
    ensure_imagenet100_root(DATA_ROOT)  # itt nincs auto-download
    train_ds, val_ds, test_ds, class_names, num_classes = load_from_directory(
        root, image_size=image_size, batch_size=batch_size, validation_split=None
    )
    return train_ds, val_ds, num_classes


KAGGLE_TARGET_PARENT = "../data/imagenet_subsets"

def load_imagenet100_kaggle(batch_size=64, image_size=(224,224)):
    root = ensure_imagenet100_from_kaggle(KAGGLE_TARGET_PARENT)  # letölt + kiépít
    ensure_imagenet100_root(KAGGLE_TARGET_PARENT)                # ellenőrzés

    train_ds, val_ds, test_ds, class_names, num_classes = load_from_directory(
        root, image_size=image_size, batch_size=batch_size, validation_split=None
    )
    return train_ds, val_ds, num_classes

def load_imagenet100_kaggle_budget(pct=0.1, batch_size=64, image_size=(224, 224), seed=42):
    """
    ImageNet-100 (Kaggle) budget loader: a train-ből osztályonként ~pct arányt mintáz,
    a val teljes marad. Ugyanaz a preprocess/batch/prefetch cső, mint a full loadernél.
    """
    # 1) biztosítsuk, hogy le legyen töltve és fel legyen építve a célstruktúra
    root = ensure_imagenet100_from_kaggle(KAGGLE_TARGET_PARENT)
    ensure_imagenet100_root(KAGGLE_TARGET_PARENT)

    train_dir = os.path.join(root, "train")

    # 2) osztálylista a train alapján (FONTOS: ugyanebben a sorrendben tanul/értékel)
    class_names = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])
    num_classes = len(class_names)

    # 3) fájllistás mintavételezés (duplikáció-védelem benne van)
    filepaths, labels = per_class_file_subset(
        train_dir, class_names, fraction=float(pct), seed=seed
    )

    # 4) subset dataset ugyanazzal a csővel (decode_image -> _standard_preprocess -> _build_ds)
    train_sub = build_dataset_from_files(
        filepaths, labels,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True
    )

    # 5) validation ugyanúgy, mint a full loadernél
    _, val_ds, _, _, _ = load_from_directory(
        root, image_size=image_size, batch_size=batch_size, validation_split=None
    )

    return train_sub, val_ds